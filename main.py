import base64
import io
import math
import platform
import sys
import numpy as np
import flet as ft
from PIL import Image, ImageDraw, ImageFont
import asyncio

# -----------------------------------------------------------------------------
# 跨平台裝置判斷
# -----------------------------------------------------------------------------
def is_mobile():
    """判斷當前是否在手機/行動裝置上執行"""
    if hasattr(sys, "getandroidapilevel") or "android" in sys.platform.lower():
        # 根據是否為手機動態調整字型比例
        return True
    if sys.platform in ["ios", "darwin"] and (
        "iPhone" in platform.machine() or "iPad" in platform.machine()
    ):
        return True
    return False

def check_is_mobile(page: ft.Page) -> bool:
    # 1. 優先使用 Flet 內建的平台判斷
    if page.platform in [ft.PagePlatform.WINDOWS]:
            return False
    
    return True
    
    #if page.platform in [ft.PagePlatform.ANDROID, ft.PagePlatform.IOS]:
    #    return True
    
    # 2. 備用：若使用 Web/PWA 模式開在手機瀏覽器，可依螢幕寬度輔助判斷
    #if page.width and page.width < 600:
    #    return True
        
    #return False

# -----------------------------------------------------------------------------
# 字型載入
# -----------------------------------------------------------------------------
def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        # 當找不到 arial.ttf 時，改用 PIL 的預設字型
        try:
            return ImageFont.load_default(size=size)  # Pillow 10.1+ 支援 size 參數
        except TypeError:
            return ImageFont.load_default()  # 舊版 Pillow 相容備案
    
    

# 標記電壓選項
VOLTAGE_OPTIONS = [
    ("161", 161000.0),
    ("22.8", 22800.0),
    ("11.4", 11400.0),
    ("6.6", 6600.0),
    ("4.16", 4160.0),
    ("3.3", 3300.0),
    ("0.46", 460.0),
    ("0.38", 380.0),
    ("0.22", 220.0),
]

CURVE_FAMILY_MAP = {
    "IEC": ["NI", "VI", "EI", "LTI"],
    "IEEE": ["MI", "VI", "EI", "LTI", "LTVI", "LTEI", "STI", "STEI"],
    "IEEE2": ["MI", "NI", "VI", "EI"]
}

# 預留邊界空間 (預留足夠空間給標題與 X/Y 軸文字，防止重疊)
#LEFT_MARGIN, RIGHT_MARGIN = 0.16, 0.94
#TOP_MARGIN, BOTTOM_MARGIN = 0.78, 0.20

# 改成這樣 (增加左、下、上的留白)：
LEFT_MARGIN, RIGHT_MARGIN = 0.10, 0.96 #0.18, 0.94
TOP_MARGIN, BOTTOM_MARGIN = 0.82, 0.12 # TOP_MARGIN larger, the upper space is lesser

# 載入字體 (若無向量字體則改用預設)
# PIL 上顯示的字型
FONT_TITLE = get_font(40)
FONT_LABEL = get_font(32)
FONT_SMALL = get_font(30)

# -----------------------------------------------------------------------------
# 跳脫時間計算邏輯
# -----------------------------------------------------------------------------
def calc_trip_time(standard, curve_type, I_base, Ip_base, TMS_TD, enable_51, enable_50, inst_ip_base, inst_time):
    t = np.full_like(I_base, np.nan, dtype=float)

    if enable_51 and Ip_base > 0:
        M = I_base / Ip_base
        valid_51 = M >= 1.001

        if standard == "IEC":
            iec_params = {"NI": (0.14, 0.02), "VI": (13.5, 1.0), "EI": (80.0, 2.0), "LTI": (120.0, 1.0)}
            if curve_type in iec_params:
                A, B = iec_params[curve_type]
                t[valid_51] = TMS_TD * (A / (np.power(M[valid_51], B) - 1.0))

        elif standard in ["IEEE", "ANSI"]:
            ieee_params = {
                "MI": (0.0515, 0.1140, 0.02), "VI": (19.61, 0.4910, 2.0), "EI": (28.2, 0.1217, 2.0),
                "LTI": (0.086, 0.185, 0.02), "LTVI": (28.55, 0.712, 2.0), "LTEI": (64.07, 0.250, 2.0),
                "STI": (0.16758, 0.11858, 0.02), "STEI": (1.281, 0.005, 2.0)
            }
            if curve_type in ieee_params:
                A, B, C = ieee_params[curve_type]
                t[valid_51] = TMS_TD * ((A / (np.power(M[valid_51], C) - 1.0)) + B)

        elif standard == "IEEE2":
            ieee2_params = {
                "MI": (0.1735, 0.6791, 0.8, -0.08, 0.1271),
                "NI": (0.0274, 2.2614, 0.3, -4.1899, 9.1272),
                "VI": (0.0615, 0.7989, 0.34, -0.284, 4.0505),
                "EI": (0.0399, 0.2294, 0.5, 3.0094, 0.7222)
            }
            if curve_type in ieee2_params:
                A, B, C, D, E = ieee2_params[curve_type]
                m_val = M[valid_51]
                valid_m = m_val > C
                m_sub = m_val[valid_m] - C
                t_calc = TMS_TD * (A + (B / m_sub) + (D / (m_sub ** 2)) + (E / (m_sub ** 3)))
                idx_51 = np.where(valid_51)[0]
                t[idx_51[valid_m]] = t_calc

    if enable_50 and inst_ip_base > 0:
        inst_mask = I_base >= inst_ip_base
        if enable_51:
            t[inst_mask] = np.nanmin([t[inst_mask], np.full(np.sum(inst_mask), inst_time)], axis=0)
        else:
            t[inst_mask] = inst_time

    t[t > 10000] = np.nan
    return t

# -----------------------------------------------------------------------------
# PIL 底圖繪製引擎
# -----------------------------------------------------------------------------
def render_trip_curve_pil(stage_configs, default_colors, selected_idx=0, test_current=None, fig_w=480, fig_h=310, scale=3, pw=None, ph=None):
    draw_w = fig_w * scale
    draw_h = fig_h * scale
    
    img = Image.new("RGB", (draw_w, draw_h), "white")
    draw = ImageDraw.Draw(img)



    x_min, x_max = 10.0, 100000.0
    y_min, y_max = 0.001, 360.0

    log_x_min, log_x_max = math.log10(x_min), math.log10(x_max)
    log_y_min, log_y_max = math.log10(y_min), math.log10(y_max)

    plot_x0 = int(draw_w * LEFT_MARGIN)
    plot_x1 = int(draw_w * RIGHT_MARGIN)
    plot_y0 = int(draw_h * (1 - TOP_MARGIN))
    plot_y1 = int(draw_h * (1 - BOTTOM_MARGIN))

    def val_to_px(val_x, val_y):
        lx = math.log10(max(val_x, x_min))
        ly = math.log10(max(val_y, y_min))
        px = plot_x0 + (lx - log_x_min) / (log_x_max - log_x_min) * (plot_x1 - plot_x0)
        py = plot_y1 - (ly - log_y_min) / (log_y_max - log_y_min) * (plot_y1 - plot_y0)
        return px, py

    selected_cfg = stage_configs[selected_idx]
    v_base = selected_cfg["voltage"]
    v_base_str = f"{v_base/1000:g}kV" if v_base >= 1000 else f"{int(v_base)}V"

    # 調整標題高度，確保推高不壓頂部圖表線 (plot_y0)
    title_y_pos = max(plot_y0 - int(35 * scale), int(4 * scale))  # 減越多，文字越往上提
    
    draw.text((plot_x0, title_y_pos), "Schneider Electric (Taiwan)", fill="#000000", font=FONT_TITLE)
    
    base_v_text = f"Base Voltage : {v_base_str}"
    draw.text((plot_x1 - int(draw_w * 0.28), title_y_pos), base_v_text, fill="#D90429", font=FONT_TITLE)

    draw.rectangle([plot_x0, plot_y0, plot_x1, plot_y1], outline="#333333", width=int(1.5 * scale))

    # X 軸刻度文字
    for dec in range(1, 6):
        base_val = 10**dec
        for sub in range(1, 10):
            v = base_val * sub
            if v > x_max: break
            px, _ = val_to_px(v, y_min)
            is_major = (sub == 1)
            draw.line([(px, plot_y0), (px, plot_y1)], fill="#E0E0E0" if not is_major else "#B0BEC5", width=int(1 * scale))
            if is_major and px <= plot_x1:
                label = f"{int(v)}" if v < 1000 else f"{int(v//1000)}k"
                draw.text((px - int(8 * scale), plot_y1 + int(4 * scale)), label, fill="#333333", font=FONT_LABEL)

    # Y 軸刻度文字 (增加左移 offset 防止遮擋邊界)
    y_ticks = [0.001, 0.01, 0.1, 1, 10, 100]
    for y_val in y_ticks:
        _, py = val_to_px(x_min, y_val)
        draw.line([(plot_x0, py), (plot_x1, py)], fill="#B0BEC5", width=int(1 * scale))
        draw.text(
            (plot_x0 - int(40 * scale), py - int(6 * scale)),
            f"{y_val:g}",
            fill="#333333",
            font=FONT_LABEL,
        )

    I_base_range = np.logspace(np.log10(x_min), np.log10(x_max), 600)

    for i, config in enumerate(stage_configs):
        if not config["enable_51"] and not config["enable_50"]:
            continue

        ratio = config["voltage"] / v_base
        ip_base = config["ip"] * ratio
        inst_ip_base = config["inst_ip"] * ratio

        if not config["enable_51"] and config["enable_50"]:
            I_curve = np.array([inst_ip_base, inst_ip_base, x_max])
            t_curve = np.array([y_max, config["inst_time"], config["inst_time"]])
        else:
            I_curve = I_base_range.copy()
            t_curve = calc_trip_time(
                standard=config["std"], curve_type=config["type"], I_base=I_curve, 
                Ip_base=ip_base, TMS_TD=config["tms"], enable_51=config["enable_51"],
                enable_50=config["enable_50"], inst_ip_base=inst_ip_base, inst_time=config["inst_time"]
            )

            if config["enable_51"] and config["enable_50"] and inst_ip_base > 0:
                t_51_at_inst = calc_trip_time(
                    standard=config["std"], curve_type=config["type"], I_base=np.array([inst_ip_base]),
                    Ip_base=ip_base, TMS_TD=config["tms"], enable_51=True, enable_50=False,
                    inst_ip_base=0, inst_time=0
                )[0]
                if not np.isnan(t_51_at_inst) and t_51_at_inst > config["inst_time"]:
                    idx = np.searchsorted(I_curve, inst_ip_base)
                    I_curve = np.insert(I_curve, idx, inst_ip_base)
                    t_curve = np.insert(t_curve, idx, t_51_at_inst)

        pts = [val_to_px(ix, tx) for ix, tx in zip(I_curve, t_curve) if not np.isnan(tx) and y_min <= tx <= y_max]

        if len(pts) > 1:
            draw.line(pts, fill=default_colors[i], width=int(2.5 * scale))

    if test_current is not None and x_min <= test_current <= x_max:
        px_test, _ = val_to_px(test_current, y_min)
        y_start, y_end = plot_y0, plot_y1
        dash_len = int(4 * scale)
        for y in range(y_start, y_end, dash_len * 2):
            draw.line([(px_test, y), (px_test, min(y + dash_len, y_end))], fill="#E63946", width=int(2 * scale))

        for i, config in enumerate(stage_configs):
            if not config["enable_51"] and not config["enable_50"]:
                continue
            
            ratio = config["voltage"] / v_base
            ip_base = config["ip"] * ratio
            inst_ip_base = config["inst_ip"] * ratio

            t_val = calc_trip_time(
                standard=config["std"], curve_type=config["type"], I_base=np.array([test_current]),
                Ip_base=ip_base, TMS_TD=config["tms"], enable_51=config["enable_51"],
                enable_50=config["enable_50"], inst_ip_base=inst_ip_base, inst_time=config["inst_time"]
            )[0]

            if not np.isnan(t_val) and y_min <= t_val <= y_max:
                _, py = val_to_px(test_current, t_val)
                r = int(4 * scale)
                draw.ellipse([px_test - r, py - r, px_test + r, py + r], fill=default_colors[i], outline="white")
                draw.text((px_test + int(6 * scale), py - int(6 * scale)), f"{t_val:.3f}s", fill='black', font=FONT_SMALL)   #fill=default_colors[i]

    # 1. 組合 Current (A) 文字
    x_label = "Current (A)"
    if pw is not None and ph is not None:
        x_label += f"  (pw:{pw:.1f}, ph:{ph:.1f})"
    
    draw.text((plot_x0 + (plot_x1 - plot_x0) // 2 - int(25 * scale), plot_y1 + int(18 * scale)), x_label, fill="#333333", font=FONT_LABEL)
    #draw.text((plot_x0 + (plot_x1 - plot_x0) // 2 - int(25 * scale), plot_y1 + int(18 * scale)), "Current (A)", fill="#333333", font=FONT_LABEL)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"

# -----------------------------------------------------------------------------
# Flet 主程式
# -----------------------------------------------------------------------------
def main(page: ft.Page):
    page.title = "保護協調曲線 (Schneider Electric)"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    page.spacing = 0

    page.window.width = 530
    page.window.height = 820
    page.window.min_width = 360
    page.window.min_height = 500
    page.window.center_on_screen = True

    default_colors = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D", "#1D3557", "#8D99AE"]
    
    stage_configs = [
        {"name": "IED_1", "voltage": 161000.0, "enable_51": True,  "std": "IEC",   "type": "NI", "ip": 200,  "tms": 0.4, "enable_50": True,  "inst_ip": 1200, "inst_time": 0.03},
        {"name": "IED_2", "voltage": 22800.0,  "enable_51": True,  "std": "IEC",   "type": "NI", "ip": 800,  "tms": 0.3, "enable_50": True,  "inst_ip": 7000, "inst_time": 0.02},
        {"name": "IED_3", "voltage": 11400.0,  "enable_51": True,  "std": "IEC",   "type": "NI", "ip": 1500, "tms": 0.2, "enable_50": True,  "inst_ip": 12000, "inst_time": 0.01},
        {"name": "IED_4", "voltage": 380.0,    "enable_51": False, "std": "IEEE2", "type": "EI", "ip": 800,  "tms": 0.4, "enable_50": False, "inst_ip": 3000, "inst_time": 0.03},
        {"name": "IED_5", "voltage": 4160.0,   "enable_51": False, "std": "IEEE",  "type": "VI", "ip": 1200, "tms": 0.5, "enable_50": False, "inst_ip": 10000,"inst_time": 0.03},
        {"name": "IED_6", "voltage": 220.0,    "enable_51": False, "std": "IEEE",  "type": "VI", "ip": 2000, "tms": 0.6, "enable_50": False, "inst_ip": 15000,"inst_time": 0.03},
    ]

    current_selected_index = [0]
    chart_dim = {"w": 480, "h": 380}

    hover_I_val_text = ft.Text("", size=9, weight=ft.FontWeight.BOLD, color="#1D3557")
    hover_details_column = ft.Column(spacing=1)

    hover_card = ft.Container(
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Container(width=4),
                        ft.Text("📍 電流:", size=9, weight=ft.FontWeight.BOLD, color="#1D3557", expand=True),
                        hover_I_val_text,
                    ],
                    spacing=1,
                ),
                ft.Divider(height=1, color="#E0E0E0"),
                hover_details_column,
            ],
            spacing=2,
        ),
        padding=5,
        bgcolor="#FFFFFF",
        border=ft.Border.all(1, "#B0BEC5"),
        border_radius=5,
        shadow=ft.BoxShadow(spread_radius=1, blur_radius=4, color="black12"),
        visible=False,
        # --- 動態精準對齊設定 ---
        top=0,      # 靠頂部對齊（後續在 on_page_resize 計算）
        right=0,    # 靠右側對齊（後續在 on_page_resize 計算）
        width=130,
    )

    # 改為 CONTAIN 避免圖片被拉伸擠壓
    chart_image = ft.Image(src="", fit="fill", expand=True)
    chart_stack = ft.Stack(controls=[chart_image, hover_card],expand=True)


    # 雙擊重置功能是由ft.GestureDetector 觸發，並呼叫 chart_interactive.reset 來實現
    chart_gesture = ft.GestureDetector(
        content=chart_stack,
        on_double_tap=lambda e: page.run_task(chart_interactive.reset)
    )

    #InteractiveViewer 是 Flet 內建提供手勢縮放與位移（Pinch-to-zoom / Pan）的容器元件，
    #設定 min_scale=1.0 與 max_scale=6.0 即定義了雙指拉近拉遠的極限。
    chart_interactive = ft.InteractiveViewer(
        content=chart_gesture,
        min_scale=1.0,
        max_scale=6.0,
        boundary_margin=ft.Margin.all(10),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        expand=True,
    )

    def update_hover_card(current_A):
        if current_A is None or current_A < 10.0 or current_A > 100000.0:
            hover_card.visible = False
            return

        v_base = stage_configs[current_selected_index[0]]["voltage"]
        hover_I_val_text.value = f"{current_A:,.1f} A"
        hover_details_column.controls.clear()

        for i, config in enumerate(stage_configs):
            if not config["enable_51"] and not config["enable_50"]:
                continue

            ratio = config["voltage"] / v_base
            ip_base = config["ip"] * ratio
            inst_ip_base = config["inst_ip"] * ratio

            t_val = calc_trip_time(
                standard=config["std"], curve_type=config["type"], I_base=np.array([current_A]),
                Ip_base=ip_base, TMS_TD=config["tms"], enable_51=config["enable_51"],
                enable_50=config["enable_50"], inst_ip_base=inst_ip_base, inst_time=config["inst_time"]
            )[0]

            t_str = "不動作" if np.isnan(t_val) else f"{t_val:.3f} s"
            v_str = f"{config['voltage']/1000:.1f}kV" if config['voltage'] >= 1000 else f"{int(config['voltage'])}V"
            
            hover_details_column.controls.append(
                ft.Row(
                    controls=[
                        ft.Container(width=5, height=5, bgcolor=default_colors[i], border_radius=2.5),
                        ft.Text(f"{config['name']} ({v_str}):", size=8.5, weight=ft.FontWeight.W_500, expand=True),
                        ft.Text(t_str, size=8.5, weight=ft.FontWeight.BOLD, color="#2B2D42"),
                    ],
                    spacing=1,
                )
            )
        hover_card.visible = True

    INPUT_HEIGHT, TEXT_HEIGHT = 36, 48
    CHK_SLOT_WIDTH, TEXT_SIZE = 30, 15
    style_text_10 = ft.TextStyle(size=16)
    pad_box = ft.Padding(6, 10, 6, 10)

    tf_test_I = ft.TextField(
        label="電流(A)", dense=True, expand=True, text_size=TEXT_SIZE, 
        label_style=style_text_10, keyboard_type=ft.KeyboardType.NUMBER, 
        content_padding=pad_box, height=TEXT_HEIGHT
    )
    tf_test_result = ft.TextField(
        label="跳脫時間 (s)", value="-", read_only=True, dense=True, expand=True,
        text_size=TEXT_SIZE, label_style=style_text_10, content_padding=pad_box, height=TEXT_HEIGHT,
        text_style=ft.TextStyle(color="#E63946", weight=ft.FontWeight.BOLD)
    )

    def update_single_test_result_text(test_i):
        idx = current_selected_index[0]
        cfg = stage_configs[idx]
        v_base = cfg["voltage"]

        if test_i is None or test_i <= 0:
            tf_test_result.value = "-"
            return

        ratio = cfg["voltage"] / v_base
        t_val = calc_trip_time(
            standard=cfg["std"], curve_type=cfg["type"],
            I_base=np.array([test_i * ratio]),
            Ip_base=cfg["ip"] * ratio, TMS_TD=cfg["tms"],
            enable_51=cfg["enable_51"], enable_50=cfg["enable_50"],
            inst_ip_base=cfg["inst_ip"] * ratio, inst_time=cfg["inst_time"]
        )[0]

        tf_test_result.value = "不動作" if np.isnan(t_val) else f"{t_val:.3f}"

    def redraw_pil_chart(current_i=None, pw=None, ph=None):
        w = chart_dim["w"]
        h = chart_dim["h"]

        chart_image.width = w
        chart_image.height = h
        chart_stack.width = w
        chart_stack.height = h
        chart_interactive.width = w
        chart_interactive.height = h

        chart_image.src = render_trip_curve_pil(
            stage_configs, default_colors, 
            selected_idx=current_selected_index[0], 
            test_current=current_i,
            fig_w=w,
            fig_h=h,
            scale=3,
            pw=pw,
            ph=ph
        )
        
        

    def on_slider_change(e):
        val_I = 10 ** e.control.value
        tf_test_I.value = f"{val_I:.1f}"
        update_single_test_result_text(val_I)
        update_hover_card(val_I)
        page.update()

    def on_slider_change_end(e):
        val_I = 10 ** e.control.value
        redraw_pil_chart(current_i=val_I)
        page.update()

    test_i_slider = ft.Slider(
        min=1.0, max=5.0, value=3.0, divisions=400,
        active_color="#E63946",
        on_change=on_slider_change,
        on_change_end=on_slider_change_end,
        expand=True
    )

    def on_test_I_input_change(e):
        try:
            val = float(tf_test_I.value.strip())
            if 10 <= val <= 100000:
                test_i_slider.value = math.log10(val)
            update_single_test_result_text(val)
            update_hover_card(val)
            redraw_pil_chart(current_i=val)
        except ValueError:
            tf_test_result.value = "格式錯誤"
            redraw_pil_chart(current_i=None)
        page.update()

    tf_test_I.on_change = on_test_I_input_change

    dd_loop_select = ft.Dropdown(
        label="迴路", options=[ft.dropdown.Option(key=str(i), text=cfg["name"]) for i, cfg in enumerate(stage_configs)],
        value="0", dense=True, text_size=TEXT_SIZE, label_style=style_text_10, content_padding=pad_box, height=INPUT_HEIGHT, expand=True
    )
    dd_voltage = ft.Dropdown(
        label="電壓(kV)", options=[ft.dropdown.Option(key=str(val), text=name) for name, val in VOLTAGE_OPTIONS],
        dense=True, text_size=TEXT_SIZE, label_style=style_text_10, content_padding=pad_box, height=INPUT_HEIGHT, expand=True
    )
    chk_enable_51 = ft.Checkbox(value=True)
    dd_std = ft.Dropdown(
        label="標準", options=[ft.dropdown.Option("IEC"), ft.dropdown.Option("IEEE"), ft.dropdown.Option("IEEE2")],
        dense=True, expand=True, text_size=TEXT_SIZE, label_style=style_text_10, content_padding=pad_box, height=INPUT_HEIGHT
    )
    dd_type = ft.Dropdown(label="型態", options=[], dense=True, expand=True, text_size=TEXT_SIZE, label_style=style_text_10, content_padding=pad_box, height=INPUT_HEIGHT)
    tf_ip = ft.TextField(label="51 Ip (A)", dense=True, expand=True, text_size=TEXT_SIZE, label_style=style_text_10, keyboard_type=ft.KeyboardType.NUMBER, content_padding=pad_box, height=TEXT_HEIGHT)
    tf_tms = ft.TextField(label="TMS/TD", dense=True, expand=True, text_size=TEXT_SIZE, label_style=style_text_10, keyboard_type=ft.KeyboardType.NUMBER, content_padding=pad_box, height=TEXT_HEIGHT)

    chk_enable_50 = ft.Checkbox(value=False)
    tf_inst_ip = ft.TextField(label="50 Ip (A)", dense=True, expand=True, text_size=TEXT_SIZE, label_style=style_text_10, keyboard_type=ft.KeyboardType.NUMBER, content_padding=pad_box, height=TEXT_HEIGHT)
    tf_inst_time = ft.TextField(label="時間 (s)", dense=True, expand=True, text_size=TEXT_SIZE, label_style=style_text_10, keyboard_type=ft.KeyboardType.NUMBER, content_padding=pad_box, height=TEXT_HEIGHT)

    def update_type_options(std_val: str, current_type_val: str = None):
        available_types = CURVE_FAMILY_MAP.get(std_val, ["NI", "VI", "EI"])
        dd_type.options = [ft.dropdown.Option(t) for t in available_types]
        dd_type.value = current_type_val if current_type_val in available_types else available_types[0]

    def update_all_and_redraw(e=None):
        idx = current_selected_index[0]
        cfg = stage_configs[idx]
        try: cfg["voltage"] = float(dd_voltage.value)
        except (ValueError, TypeError): cfg["voltage"] = 161000.0
        cfg["enable_51"] = chk_enable_51.value
        cfg["std"] = dd_std.value
        cfg["type"] = dd_type.value
        try: cfg["ip"] = float(tf_ip.value)
        except ValueError: cfg["ip"] = 100.0
        try: cfg["tms"] = float(tf_tms.value)
        except ValueError: cfg["tms"] = 0.1
        cfg["enable_50"] = chk_enable_50.value
        try: cfg["inst_ip"] = float(tf_inst_ip.value)
        except ValueError: cfg["inst_ip"] = 1000.0
        try: cfg["inst_time"] = float(tf_inst_time.value)
        except ValueError: cfg["inst_time"] = 0.03

        try: cur_i = float(tf_test_I.value.strip())
        except ValueError: cur_i = None

        redraw_pil_chart(current_i=cur_i)
        if cur_i:
            update_single_test_result_text(cur_i)
            update_hover_card(cur_i)
        page.update()

    def load_loop_data(idx: int):
        cfg = stage_configs[idx]
        dd_voltage.value = str(cfg["voltage"])
        chk_enable_51.value = cfg["enable_51"]
        dd_std.value = cfg["std"]
        update_type_options(cfg["std"], cfg["type"])
        tf_ip.value = str(cfg["ip"])
        tf_tms.value = str(cfg["tms"])
        chk_enable_50.value = cfg["enable_50"]
        tf_inst_ip.value = str(cfg["inst_ip"])
        tf_inst_time.value = str(cfg["inst_time"])
        update_all_and_redraw()

    dd_loop_select.on_select = lambda e: (current_selected_index.__setitem__(0, int(e.control.value)), load_loop_data(int(e.control.value)))
    dd_voltage.on_select = update_all_and_redraw
    dd_std.on_select = lambda e: (update_type_options(dd_std.value), update_all_and_redraw(e))
    dd_type.on_select = update_all_and_redraw
    chk_enable_51.on_change = update_all_and_redraw
    tf_ip.on_change = update_all_and_redraw
    tf_tms.on_change = update_all_and_redraw
    chk_enable_50.on_change = update_all_and_redraw
    tf_inst_ip.on_change = update_all_and_redraw
    tf_inst_time.on_change = update_all_and_redraw

    test_panel = ft.Container(
        content=ft.Column(
            controls=[
                ft.Text("🎯 電流測試試算", size=11, weight=ft.FontWeight.BOLD, color="#1D3557"),
                ft.Row([tf_test_I, tf_test_result], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER, height=40),
            ],
            spacing=4,
        ),
        padding=6,
        margin=ft.Margin(left=40),  # <--- 加上這行，數值越大越往右移 (例如 20px 或 40px)
        bgcolor="#EBF3FA",
        border_radius=6,
    )

    top_chart_panel = ft.Container(
        content=ft.Column(
            controls=[
                ft.Container(
                    content=chart_interactive, 
                    alignment=ft.Alignment(0, 0),
                ),
                test_panel,
                ft.Row(
                    controls=[
                        ft.Text("10A", size=10, color="#666666"),
                        test_i_slider,
                        ft.Text("100kA", size=10, color="#666666"),
                    ],
                    spacing=2,
                    alignment=ft.MainAxisAlignment.CENTER
                )
            ],
            spacing=4,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        ),
        padding=2,
        expand=True
    )

    # 建立裝置提示文字
    #device_info_str = f"📱 手機模式 ({sys.platform})" if is_mobile() else f"💻 電腦模式 ({sys.platform})"

    # 取得是否為手機
    is_mobile_dev = check_is_mobile(page)

    # 顯示提示文字
    device_info_str = f"📱 手機模式 ({page.platform.value})" if is_mobile_dev else f"💻 電腦模式 ({page.platform.value})"

    bottom_setting_panel = ft.Container(
        content=ft.Column(
            controls=[
                # 將原本的 ft.Text 改為 ft.Row，把偵測結果放在右側
                ft.Row(
                    controls=[
                        ft.Text("⚡ 保護參數設定", size=12, weight=ft.FontWeight.BOLD),
                        ft.Text(f"[{device_info_str}]", size=10, color="#E63946", weight=ft.FontWeight.BOLD),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Row([ft.Container(width=CHK_SLOT_WIDTH), dd_loop_select, dd_voltage], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(height=1, thickness=1, color="#E0E0E0"),
                ft.Row([ft.Container(width=CHK_SLOT_WIDTH, content=chk_enable_51, alignment=ft.Alignment(0, 0)), dd_std, dd_type], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Row([ft.Container(width=CHK_SLOT_WIDTH), tf_ip, tf_tms], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER, height=40),
                ft.Divider(height=1, thickness=1, color="#E0E0E0"),
                ft.Row([ft.Container(width=CHK_SLOT_WIDTH, content=chk_enable_50, alignment=ft.Alignment(0, 0)), tf_inst_ip, tf_inst_time], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER, height=40),
            ],
            spacing=8,
        ),
        padding=8,
        border=ft.Border.all(1, "#DDDDDD"),
        border_radius=8,
        bgcolor="#FAFAFA",
        expand=True
    )

    def on_page_resize(e):
            # 優先從事件 e 取得 width / height，若無則讀取 page 的寬高
            if e and hasattr(e, "width") and e.width > 0:
                pw = e.width
                ph = e.height
            else:
                pw = page.width if (page.width and page.width > 0) else 360
                ph = page.height if (page.height and page.height > 0) else 800
    
            is_landscape = pw > ph

            # 橫向與縱向寬度計算
            new_w = max(int(pw - 12), 300)

            if is_landscape:
                new_h = max(int(ph * 0.48), 180)
            else:
                new_h = min(int(new_w / 1.35), 300)

            chart_dim["w"] = new_w
            chart_dim["h"] = new_h
            
            # -------------------------------------------------------------
            # 動態計算浮動卡片的位置 (對齊 PIL 網格右上角)
            # -------------------------------------------------------------
            # PIL 的頂部邊界為 (1 - TOP_MARGIN)，即 1 - 0.82 = 0.18
            # PIL 的右側邊界為 RIGHT_MARGIN，即 0.96 (右邊留白 4%)
            
            offset_top = int(new_h * (1 - TOP_MARGIN)) + 2   # +2px 避免壓到圖表上邊框
            offset_right = int(new_w * (1 - RIGHT_MARGIN)) + 2 # +2px 避免壓到圖表右邊框

            hover_card.top = offset_top
            hover_card.right = offset_right
            

            # 重新繪製 PIL 圖表
            try:
                cur_i = float(tf_test_I.value.strip())
            except (ValueError, AttributeError):
                cur_i = None

            redraw_pil_chart(current_i=cur_i, pw=pw, ph=ph)
            page.update()


    # 建立主容器，並將 on_resize 綁定在此容器上
    main_container = ft.Container(
        content=ft.Column(
            controls=[
                top_chart_panel,
                bottom_setting_panel,
            ],
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            scroll=ft.ScrollMode.AUTO,
        ),
        padding=ft.Padding(left=6, right=6, top=6, bottom=12),
    )

    page.on_resized = on_page_resize
    
    page.add(
            ft.SafeArea(
                main_container
            )
        )

    load_loop_data(0)
    on_page_resize(None)
    
    # 加入背景尺寸監測 (解決 Mobile 轉向事件丟失的問題)
    async def monitor_orientation():
        last_w, last_h = page.width, page.height
        while True:
            await asyncio.sleep(0.5)
            if page.width != last_w or page.height != last_h:
                last_w, last_h = page.width, page.height
                on_page_resize(None)

    page.run_task(monitor_orientation)
    
if __name__ == "__main__":
    ft.run(main)