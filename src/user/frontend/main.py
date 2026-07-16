import asyncio
import uuid
import json
import logging
import flet as ft
import flet.canvas as cv

from src.contracts import block_schemas
from src.user.frontend.api_client import ShivaApiClient
from src.user.frontend.config import load_config
from src.user.frontend.workflow_builder import build_workflow_request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shiva.frontend")

# Card properties
CARD_WIDTH = 160
CARD_HEIGHT = 80

# Frontend-only visual treatment for contract block schemas.
BLOCK_VISUALS = {
    "camera": {
        "gradient_colors": [ft.Colors.TEAL_600, ft.Colors.GREEN_500],
        "icon": ft.Icons.CAMERA_ALT,
    },
    "microphone": {
        "gradient_colors": [ft.Colors.BLUE_600, ft.Colors.CYAN_500],
        "icon": ft.Icons.MIC,
    },
    "network": {
        "gradient_colors": [ft.Colors.ORANGE_600, ft.Colors.AMBER_500],
        "icon": ft.Icons.LANGUAGE,
    },
    "prompt": {
        "gradient_colors": [ft.Colors.AMBER_500, ft.Colors.YELLOW_600],
        "icon": ft.Icons.EDIT_NOTE,
    },
    "if_else": {
        "gradient_colors": [ft.Colors.PURPLE_600, ft.Colors.PINK_500],
        "icon": ft.Icons.CALL_SPLIT,
    },
    "or": {
        "gradient_colors": [ft.Colors.PURPLE_500, ft.Colors.DEEP_PURPLE_600],
        "icon": ft.Icons.MERGE,
    },
    "and": {
        "gradient_colors": [ft.Colors.PURPLE_700, ft.Colors.INDIGO_600],
        "icon": ft.Icons.ALT_ROUTE,
    },
    "shiva_output": {
        "gradient_colors": [ft.Colors.RED_600, ft.Colors.PINK_600],
        "icon": ft.Icons.OUTPUT,
    },
}


def build_block_definitions(schema_payloads):
    definitions = {}
    for schema in schema_payloads:
        block_type = schema["block_type"]
        definitions[block_type] = {
            **schema,
            **BLOCK_VISUALS.get(
                block_type,
                {
                    "gradient_colors": [ft.Colors.GREY_700, ft.Colors.GREY_500],
                    "icon": ft.Icons.EXTENSION,
                },
            ),
        }
    return definitions


BLOCK_DEFINITIONS = build_block_definitions([schema.to_json() for schema in block_schemas()])

# Global State
blocks = {}
edges = []
selected_output_block_id = None
canvas_stack = None
connections_canvas = None
instruction_input = None
output_console = None
page_ref = None

async def main(page: ft.Page):
    global canvas_stack, connections_canvas, instruction_input, output_console, page_ref, BLOCK_DEFINITIONS
    page_ref = page
    config = load_config()
    api_client = ShivaApiClient(config)

    try:
        BLOCK_DEFINITIONS = build_block_definitions(api_client.block_schemas())
        logger.info("Loaded block schemas from backend.")
    except Exception as exc:
        logger.info(f"Using local block schemas; backend schemas unavailable: {exc}")
    
    # Configure native window settings
    page.title = "Shiva.ai Flow - Workflow Studio"
    page.bgcolor = "#08080c"
    page.padding = 15
    page.window.width = 1100
    page.window.height = 800
    page.window.min_width = 800
    page.window.min_height = 600
    
    # Define custom UI style components
    page.fonts = {
        "Outfit": "https://github.com/google/fonts/raw/main/ofl/outfit/Outfit%5Bwght%5D.ttf"
    }
    page.theme = ft.Theme(font_family="Outfit")
    
    # Setup notifications / toasts helper
    def show_toast(message: str, is_error: bool = False):
        page.overlay.append(
            ft.SnackBar(
                content=ft.Text(message, color=ft.Colors.WHITE),
                bgcolor=ft.Colors.RED_700 if is_error else ft.Colors.GREEN_700,
                show_close_icon=True
            )
        )
        page.update()

    # Sockets interaction handlers
    def on_output_socket_click(block_id: str):
        global selected_output_block_id
        if selected_output_block_id == block_id:
            # Deselect if clicked again
            selected_output_block_id = None
        else:
            selected_output_block_id = block_id
            
        update_socket_highlights()

    def on_input_socket_click(block_id: str):
        global selected_output_block_id
        if selected_output_block_id is not None:
            if selected_output_block_id != block_id:
                # Add edge connection
                exists = any(
                    e["source_id"] == selected_output_block_id and e["target_id"] == block_id
                    for e in edges
                )
                if not exists:
                    edges.append({
                        "id": f"edge_{uuid.uuid4().hex[:6]}",
                        "source_id": selected_output_block_id,
                        "target_id": block_id
                    })
                    selected_output_block_id = None
                    update_socket_highlights()
                    redraw_connections()
                else:
                    show_toast("Connection already exists!", is_error=True)
            else:
                show_toast("Cannot connect block to itself!", is_error=True)

    def update_socket_highlights():
        for bid, b in blocks.items():
            if "output_socket_control" in b:
                socket_container = b["output_socket_control"].content
                if selected_output_block_id == bid:
                    socket_container.bgcolor = ft.Colors.GREEN_400
                    socket_container.border = ft.border.all(2, ft.Colors.WHITE)
                else:
                    socket_container.bgcolor = ft.Colors.WHITE
                    socket_container.border = ft.border.all(2, ft.Colors.BLACK)
        page.update()

    # Block movements / dragging
    def on_block_pan(e: ft.DragUpdateEvent, block_id: str):
        block = blocks[block_id]
        block["x"] += e.delta_x
        block["y"] += e.delta_y
        
        # Keep inside canvas boundaries roughly
        block["x"] = max(10, min(block["x"], 1500))
        block["y"] = max(10, min(block["y"], 800))
        
        # Update UI control coordinates
        block["control"].left = block["x"]
        block["control"].top = block["y"]
        page.update()
        
        # Redraw connection paths
        redraw_connections()

    # Drawing connection lines
    def redraw_connections():
        connections_canvas.shapes.clear()
        
        # Redraw a grid of background dots to make the canvas look premium
        # Draw dynamic dots across an arbitrary 1800x1000 canvas area
        for gx in range(20, 1800, 40):
            for gy in range(20, 1000, 40):
                connections_canvas.shapes.append(
                    cv.Circle(
                        x=gx,
                        y=gy,
                        radius=1,
                        paint=ft.Paint(color="#1a1a24")
                    )
                )

        # Draw active node connections
        for edge in edges:
            source = blocks.get(edge["source_id"])
            target = blocks.get(edge["target_id"])
            
            if source and target:
                # Output port coordinates (center of the right edge of source card)
                x1 = source["x"] + CARD_WIDTH
                y1 = source["y"] + CARD_HEIGHT / 2
                
                # Input port coordinates (center of the left edge of target card)
                x2 = target["x"]
                y2 = target["y"] + CARD_HEIGHT / 2
                
                # Draw a nice clean connection line matching the mockup
                connections_canvas.shapes.append(
                    cv.Line(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        paint=ft.Paint(
                            color=ft.Colors.WHITE,
                            stroke_width=3,
                            stroke_cap=ft.StrokeCap.ROUND
                        )
                    )
                )
        connections_canvas.update()

    # Block deletion
    def delete_block(block_id: str):
        global selected_output_block_id, edges
        if selected_output_block_id == block_id:
            selected_output_block_id = None
            
        # Clean up related connections
        edges = [e for e in edges if e["source_id"] != block_id and e["target_id"] != block_id]
        
        # Remove positioned control
        control = blocks[block_id]["control"]
        canvas_stack.controls.remove(control)
        
        # Clean up memory state
        del blocks[block_id]
        
        redraw_connections()
        page.update()
        show_toast(f"Deleted block: {block_id}")

    # Modal block configurations dialog
    def edit_block_config(block_id: str):
        block = blocks[block_id]
        block_def = BLOCK_DEFINITIONS[block["type"]]
        fields = []
        inputs_dict = {}

        for field_schema in block_def.get("fields", []):
            field_name = field_schema["name"]
            if block_def["category"] == "decision" and field_name in block.get("condition", {}):
                value = block["condition"].get(field_name, field_schema.get("default"))
            else:
                value = block["arguments"].get(field_name, field_schema.get("default"))

            if value is None:
                value = "None"

            if field_schema.get("field_type") == "select":
                control = ft.Dropdown(
                    label=field_schema["label"],
                    options=[ft.dropdown.Option(option) for option in field_schema.get("options", [])],
                    value=str(value),
                    border_color="#3a3a4c",
                )
            else:
                control = ft.TextField(
                    label=field_schema["label"],
                    value=str(value),
                    keyboard_type=ft.KeyboardType.NUMBER if field_schema.get("field_type") == "number" else None,
                    text_size=14,
                    border_color="#3a3a4c",
                )

            inputs_dict[field_name] = (control, field_schema)
            fields.append(control)

        def on_save_click(e):
            for field_name, (control, field_schema) in inputs_dict.items():
                raw_value = control.value
                if isinstance(raw_value, str) and raw_value.lower() in ["none", "null", ""]:
                    value = None
                elif field_schema.get("field_type") == "number":
                    try:
                        value = int(raw_value)
                    except (TypeError, ValueError):
                        value = field_schema.get("default", 0)
                else:
                    value = raw_value

                if block_def["category"] == "decision" and field_name in block.get("condition", {}):
                    block["condition"][field_name] = value
                else:
                    block["arguments"][field_name] = value
                
            page.dialog.open = False
            page.update()
            show_toast(f"Saved configuration for {block_id}")

        page.dialog = ft.AlertDialog(
            title=ft.Text(f"Configure {block['name']}", size=18, weight=ft.FontWeight.BOLD),
            content=ft.Container(
                content=ft.Column(fields, tight=True, spacing=12),
                width=320
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: dismiss_dialog()),
                ft.ElevatedButton("Save", bgcolor=ft.Colors.GREY_800, color=ft.Colors.WHITE, on_click=on_save_click)
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            bgcolor="#111116",
            shape=ft.RoundedRectangleBorder(radius=12)
        )
        page.dialog.open = True
        page.update()

    def dismiss_dialog():
        page.dialog.open = False
        page.update()

    # Dynamic Block instantiation on Canvas
    def create_block_on_canvas(block_id: str):
        block = blocks[block_id]
        block_def = BLOCK_DEFINITIONS[block["type"]]
        
        sockets = []
        
        # Left Input Socket (Decisions and Outputs only)
        if block_def["category"] in ["decision", "output"]:
            input_socket = ft.GestureDetector(
                mouse_cursor=ft.MouseCursor.CLICK,
                on_tap=lambda e, bid=block_id: on_input_socket_click(bid),
                content=ft.Container(
                    width=12,
                    height=12,
                    shape=ft.BoxShape.CIRCLE,
                    bgcolor=ft.Colors.WHITE,
                    border=ft.border.all(1.5, ft.Colors.BLACK),
                ),
                left=-6,
                top=CARD_HEIGHT / 2 - 6
            )
            sockets.append(input_socket)
            
        # Right Output Socket (Inputs and Decisions only)
        if block_def["category"] in ["input", "decision"]:
            output_socket = ft.GestureDetector(
                mouse_cursor=ft.MouseCursor.CLICK,
                on_tap=lambda e, bid=block_id: on_output_socket_click(bid),
                content=ft.Container(
                    width=12,
                    height=12,
                    shape=ft.BoxShape.CIRCLE,
                    bgcolor=ft.Colors.WHITE,
                    border=ft.border.all(1.5, ft.Colors.BLACK),
                ),
                left=CARD_WIDTH - 6,
                top=CARD_HEIGHT / 2 - 6
            )
            sockets.append(output_socket)
            block["output_socket_control"] = output_socket

        # Main Block Container representation
        card = ft.GestureDetector(
            on_pan_update=lambda e, bid=block_id: on_block_pan(e, bid),
            on_double_tap=lambda e, bid=block_id: edit_block_config(bid),
            content=ft.Container(
                width=CARD_WIDTH,
                height=CARD_HEIGHT,
                gradient=ft.LinearGradient(
                    begin=ft.alignment.top_left,
                    end=ft.alignment.bottom_right,
                    colors=block_def["gradient_colors"]
                ),
                border_radius=10,
                padding=10,
                shadow=ft.BoxShadow(
                    blur_radius=6,
                    color=ft.Colors.with_opacity(0.4, ft.Colors.BLACK),
                    offset=ft.Offset(2, 2)
                ),
                content=ft.Column([
                    ft.Row([
                        ft.Row([
                            ft.Icon(block_def["icon"], size=16, color=ft.Colors.WHITE),
                            ft.Text(block["name"], size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                        ], spacing=4),
                        ft.IconButton(
                            icon=ft.Icons.CLOSE,
                            icon_size=12,
                            icon_color=ft.Colors.WHITE_70,
                            padding=0,
                            on_click=lambda e, bid=block_id: delete_block(bid)
                        )
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Text(f"ID: {block_id}", size=9, color=ft.Colors.WHITE_60),
                    ft.Row([
                        ft.Text("Double-click to config", size=8.5, italic=True, color=ft.Colors.WHITE_60),
                        ft.IconButton(
                            icon=ft.Icons.SETTINGS,
                            icon_size=12,
                            icon_color=ft.Colors.WHITE,
                            padding=0,
                            on_click=lambda e, bid=block_id: edit_block_config(bid)
                        )
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
                ], spacing=2)
            )
        )

        block_stack = ft.Stack(
            controls=[card] + sockets,
            width=CARD_WIDTH,
            height=CARD_HEIGHT
        )
        
        positioned = ft.Positioned(
            left=block["x"],
            top=block["y"],
            content=block_stack
        )
        
        block["control"] = positioned
        canvas_stack.controls.append(positioned)
        page.update()

    def add_block_type(block_type: str):
        # Create block instance
        block_id = f"{block_type}_{uuid.uuid4().hex[:4]}"
        block_def = BLOCK_DEFINITIONS[block_type]
        
        # Render roughly staggered in canvas center
        x = 250.0 + len(blocks) * 30.0
        y = 180.0 + len(blocks) * 20.0
        
        blocks[block_id] = {
            "id": block_id,
            "type": block_type,
            "name": block_def["name"],
            "x": x,
            "y": y,
            "arguments": dict(block_def["default_arguments"]),
            "condition": dict(block_def.get("condition", {})),
            "control": None
        }
        
        create_block_on_canvas(block_id)
        page.end_drawer.open = False
        page.update()
        redraw_connections()
        show_toast(f"Added {block_def['name']}")

    # Helper to construct side drawer listing contract block schemas
    def make_sidebar_tile(name: str, block_type: str, icon, color):
        return ft.Container(
            content=ft.Row([
                ft.Icon(icon, size=18, color=color),
                ft.Text(name, size=14, color=ft.Colors.WHITE, weight=ft.FontWeight.W_500),
            ], spacing=10),
            padding=ft.padding.symmetric(10, 12),
            bgcolor="#161622",
            border_radius=8,
            on_click=lambda e, bt=block_type: add_block_type(bt),
            mouse_cursor=ft.MouseCursor.CLICK,
        )

    def sidebar_controls():
        controls = [
            ft.Text("Available Blocks", size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
            ft.Divider(color=ft.Colors.WHITE24, height=15),
        ]
        category_labels = [
            ("input", "INPUTS"),
            ("decision", "DECISIONS"),
            ("output", "OUTPUTS"),
        ]
        for category, label in category_labels:
            category_blocks = [
                definition
                for definition in BLOCK_DEFINITIONS.values()
                if definition["category"] == category
            ]
            if not category_blocks:
                continue
            if len(controls) > 2:
                controls.append(ft.Divider(color=ft.Colors.WHITE10, height=10))
            controls.append(ft.Text(label, size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_500))
            for definition in category_blocks:
                controls.append(
                    make_sidebar_tile(
                        definition["name"],
                        definition["block_type"],
                        definition["icon"],
                        definition["gradient_colors"][0],
                    )
                )
        return controls

    page.end_drawer = ft.NavigationDrawer(
        controls=[
            ft.Container(
                padding=20,
                content=ft.Column(sidebar_controls(), spacing=8, scroll=ft.ScrollMode.ADAPTIVE)
            )
        ],
        bgcolor="#0c0c12"
    )

    def show_side_drawer(e):
        page.end_drawer.open = True
        page.update()

    def validate_and_serialize():
        request = build_workflow_request(
            blocks=blocks,
            edges=edges,
            block_definitions=BLOCK_DEFINITIONS,
            query=instruction_input.value,
        )
        return request.to_json()

    # Log/Execution Console functions
    def append_console_log(text: str, is_bold: bool = False, color: str = ft.Colors.WHITE):
        output_console.controls.append(
            ft.Text(text, size=13, font_family="Courier New", weight=ft.FontWeight.BOLD if is_bold else ft.FontWeight.NORMAL, color=color)
        )
        output_console.update()

    async def execute_apply_workflow(e):
        append_console_log("\n>>> COMPILING WORKFLOW GRAPH...", is_bold=True, color=ft.Colors.BLUE_400)
        try:
            payload = validate_and_serialize()
            append_console_log(f"Successfully compiled graph topological sort order.", color=ft.Colors.GREEN_400)
            append_console_log(f"Payload:\n{json.dumps(payload, indent=2)}")
        except Exception as err:
            show_toast(str(err), is_error=True)
            append_console_log(f"COMPILE ERROR: {str(err)}", is_bold=True, color=ft.Colors.RED_400)
            return

        append_console_log(f"\n>>> INITIATING CONNECTION TO LOCAL RUNTIME SERVER ({api_client.base_url})...", is_bold=True, color=ft.Colors.BLUE_400)
        
        try:
            api_client.health()
            append_console_log("Server status: ONLINE. Executing workflow request...", color=ft.Colors.GREEN_400)
            res_json = api_client.execute_workflow(payload)
            append_console_log("\n>>> WORKFLOW EXECUTION COMPLETE", is_bold=True, color=ft.Colors.GREEN_400)
            append_console_log(f"Result Format: {res_json.get('format', 'text')}", is_bold=True)
            append_console_log(f"Response:\n{res_json.get('text', '')}", color=ft.Colors.LIGHT_GREEN_accent)
        except Exception as exc:
            logger.warning(f"Failed to connect to local orchestrator server: {exc}")
            if not config.simulation_fallback_enabled:
                append_console_log(f"EXECUTION ERROR: {exc}", is_bold=True, color=ft.Colors.RED_400)
                show_toast("Backend execution failed.", is_error=True)
                return
            append_console_log(f"Server status: OFFLINE. Initializing simulated cortical loop...", color=ft.Colors.ORANGE_400)
            
            # Simulated Execution sequence (matching backend WS statuses)
            steps = [
                ("starting", "Initiating workflow execution sequence.", 0.8),
                ("capturing", "Triggering hardware perception sensors.", 1.2),
                ("reasoning", "Orchestrating cortical columns (Swarm reasoning cycles active).", 1.5),
                ("completed", "Swarm successfully reached decision state.", 0.5)
            ]
            
            for status, message, delay in steps:
                append_console_log(f"[{status.upper()}] {message}", color=ft.Colors.AMBER_400)
                await asyncio.sleep(delay)
                
            append_console_log("\n>>> WORKFLOW EXECUTION COMPLETE (SIMULATED)", is_bold=True, color=ft.Colors.GREEN_400)
            
            # Formulate simulated answer based on instructions
            mock_ans = f"Simulated Decision Result based on inputs: Successful execution of '{payload['query']}'."
            append_console_log(f"Response:\n{mock_ans}", color=ft.Colors.LIGHT_GREEN_accent)
            show_toast("Executed in Simulation Mode (Backend Offline). Start server using 'uvicorn src.orchestrator.server:app --reload'")

    def clear_canvas(e):
        global selected_output_block_id, edges
        selected_output_block_id = None
        edges.clear()
        blocks.clear()
        canvas_stack.controls.clear()
        
        # Re-add canvas background and console log
        canvas_stack.controls.append(connections_canvas)
        canvas_stack.controls.append(
            ft.Positioned(
                right=15,
                top=15,
                content=ft.Container(
                    content=output_console,
                    width=380,
                    height=280,
                    bgcolor="#0d0d14",
                    border=ft.border.all(1.5, "#222230"),
                    border_radius=8,
                    padding=10
                )
            )
        )
        
        instruction_input.value = ""
        output_console.controls.clear()
        append_console_log("Canvas and console reset. Add blocks to build a new workflow.", color=ft.Colors.WHITE_50)
        
        redraw_connections()
        page.update()
        show_toast("Canvas cleared")

    # Header Row
    header = ft.Row([
        ft.Text("Flow", size=32, weight=ft.FontWeight.W_800, color=ft.Colors.WHITE),
        ft.Spacer(),
        ft.IconButton(
            icon=ft.Icons.MENU, 
            icon_size=28, 
            icon_color=ft.Colors.WHITE, 
            on_click=show_side_drawer,
            tooltip="Open Block Library"
        )
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
    
    # Subheader Action pills
    apply_button = ft.Container(
        content=ft.Text("Apply", color=ft.Colors.WHITE, size=13, weight=ft.FontWeight.BOLD),
        bgcolor="#1c1c24",
        border_radius=20,
        padding=ft.padding.symmetric(6, 16),
        on_click=execute_apply_workflow,
        mouse_cursor=ft.MouseCursor.CLICK,
    )
    
    clear_button = ft.Container(
        content=ft.Text("Clear", color=ft.Colors.WHITE, size=13, weight=ft.FontWeight.BOLD),
        bgcolor="#1c1c24",
        border_radius=20,
        padding=ft.padding.symmetric(6, 16),
        on_click=clear_canvas,
        mouse_cursor=ft.MouseCursor.CLICK,
    )
    
    sub_header = ft.Container(
        content=ft.Row([apply_button, clear_button], spacing=10),
        border=ft.border.symmetric(vertical=ft.BorderSide(1.5, ft.Colors.WHITE24)),
        padding=ft.padding.symmetric(10, 0),
        margin=ft.margin.symmetric(10, 0)
    )

    # Canvas and Output console panel setup
    connections_canvas = cv.Canvas(expand=True)
    
    output_console = ft.Column(
        controls=[ft.Text("Console log initialized. Add blocks to build a workflow.", size=13, font_family="Courier New", color=ft.Colors.WHITE_50)],
        scroll=ft.ScrollMode.ALWAYS,
        expand=True
    )
    
    console_box = ft.Positioned(
        right=15,
        top=15,
        content=ft.Container(
            content=output_console,
            width=380,
            height=280,
            bgcolor="#0d0d14",
            border=ft.border.all(1.5, "#222230"),
            border_radius=8,
            padding=10
        )
    )
    
    canvas_stack = ft.Stack(
        controls=[connections_canvas, console_box],
        expand=True
    )
    
    canvas_container = ft.Container(
        content=canvas_stack,
        bgcolor="#07070a",
        border=ft.border.all(1.5, ft.Colors.WHITE10),
        border_radius=12,
        expand=True,
        padding=0
    )

    # Bottom Instructions Area
    instruction_input = ft.TextField(
        hint_text="Please Enter your instruction...",
        hint_style=ft.TextStyle(color=ft.Colors.WHITE_38, size=15),
        multiline=True,
        min_lines=3,
        max_lines=3,
        border=ft.InputBorder.NONE,
        text_style=ft.TextStyle(color=ft.Colors.WHITE, size=15),
        content_padding=0
    )
    
    bottom_panel = ft.Container(
        content=instruction_input,
        bgcolor="#111116",
        border_radius=12,
        padding=15,
        border=ft.border.all(1, "#20202a")
    )

    # Add components to Page
    page.add(
        ft.Column([
            header,
            sub_header,
            canvas_container,
            bottom_panel
        ], expand=True, spacing=0)
    )
    
    # Pre-populate mockup layout nodes automatically to impress
    # Yellow card: Input Block 2 (Prompt) -> Purple: Instruction (If Else)
    # Blue card: Input Block 1 (Mic) -> Purple: Instruction (If Else)
    # This matches the user's mockup precisely!
    
    b1_id = "microphone_b1"
    blocks[b1_id] = {
        "id": b1_id,
        "type": "microphone",
        "name": "Input block 1",
        "x": 100.0,
        "y": 300.0,
        "arguments": dict(BLOCK_DEFINITIONS["microphone"]["default_arguments"]),
        "control": None
    }
    
    b2_id = "prompt_b2"
    blocks[b2_id] = {
        "id": b2_id,
        "type": "prompt",
        "name": "Input block 2",
        "x": 100.0,
        "y": 140.0,
        "arguments": dict(BLOCK_DEFINITIONS["prompt"]["default_arguments"]),
        "control": None
    }
    
    dec_id = "ifelse_dec"
    blocks[dec_id] = {
        "id": dec_id,
        "type": "if_else",
        "name": "(Instruction Block)",
        "x": 420.0,
        "y": 210.0,
        "arguments": dict(BLOCK_DEFINITIONS["if_else"]["default_arguments"]),
        "condition": dict(BLOCK_DEFINITIONS["if_else"]["condition"]),
        "control": None
    }
    
    # Create them on canvas
    create_block_on_canvas(b2_id)
    create_block_on_canvas(b1_id)
    create_block_on_canvas(dec_id)
    
    # Add connection edges matching mockup
    edges.append({"id": "edge_m1", "source_id": b2_id, "target_id": dec_id})
    edges.append({"id": "edge_m2", "source_id": b1_id, "target_id": dec_id})
    
    redraw_connections()

if __name__ == "__main__":
    ft.app(target=main)
