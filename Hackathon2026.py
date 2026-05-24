import os
import sys
import random
import cv2
import requests
import numpy as np
from datetime import datetime

# --- AI INTEGRATION ---
import tensorflow as tf
# Using MobileNetV2: lightweight and efficient for real-time apps
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Core System Fix for macOS System Library Discovery
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")

from kivy.lang import Builder
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.behaviors import ButtonBehavior

from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.pickers import MDDatePicker
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.card import MDCard
from kivymd.uix.textfield import MDTextField

KV = '''
MDScreenManager:
    HomeScreen:
    ScannerScreen:
    DashboardScreen:
    RecipeScreen:

<HomeScreen>:
    name: "home_screen"
   
    MDBoxLayout:
        orientation: 'vertical'
        padding: "24dp"
        spacing: "24dp"
       
        MDLabel:
            text: "NutriScan AI"
            font_style: "H4"
            halign: "center"
            theme_text_color: "Custom"
            text_color: 46/255, 125/255, 50/255, 1
            size_hint_y: None
            height: "60dp"

        MDBoxLayout:
            orientation: 'vertical'
            spacing: "16dp"
            pos_hint: {"center_x": .5, "center_y": .5}
            adaptive_height: True
            width: "280dp"
            size_hint_x: None

            MDRaisedButton:
                text: "Scan Barcode"
                size_hint_x: 1
                on_release:
                    root.manager.current = "scanner_screen"
                    app.start_camera()

            MDRaisedButton:
                text: "Manual Entry"
                size_hint_x: 1
                md_bg_color: 46/255, 125/255, 50/255, 1
                on_release: app.open_manual_entry_flow()

            MDRaisedButton:
                text: "View Dashboard & History"
                size_hint_x: 1
                on_release: app.view_dashboard()
                md_bg_color: 25/255, 118/255, 210/255, 1

            MDRaisedButton:
                text: "Explore Recipes"
                size_hint_x: 1
                on_release:
                    root.manager.current = "recipe_screen"
                    app.reload_recipes()
                md_bg_color: 230/255, 124/255, 115/255, 1

<ScannerScreen>:
    name: "scanner_screen"
   
    MDFloatLayout:
        MDBoxLayout:
            orientation: 'vertical'
            padding: "16dp"
            spacing: "16dp"
            pos_hint: {"x": 0, "y": 0}
            size_hint: 1, 1

            MDBoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: "50dp"
                spacing: "10dp"

                MDRaisedButton:
                    text: "← Back"
                    on_release:
                        app.stop_camera()
                        root.manager.current = "home_screen"
               
                MDRaisedButton:
                    text: "AI Identify Food"
                    md_bg_color: 103/255, 58/255, 183/255, 1
                    on_release: app.run_ai_recognition()

            Image:
                id: camera_preview
                allow_stretch: True
                keep_ratio: True

            MDLabel:
                id: status_label
                text: "Align barcode or tap 'AI Identify'"
                halign: "center"
                size_hint_y: None
                height: "40dp"

<DashboardScreen>:
    name: "dashboard_screen"
   
    MDBoxLayout:
        orientation: 'vertical'
        padding: "24dp"
        spacing: "20dp"
       
        MDRaisedButton:
            text: "← Back to Home"
            on_release: root.manager.current = "home_screen"
           
        MDLabel:
            text: "Inventory & Nutrient Hub"
            font_style: "H5"
            halign: "center"
            theme_text_color: "Custom"
            text_color: 46/255, 125/255, 50/255, 1
            size_hint_y: None
            height: "50dp"
           
        ScrollView:
            do_scroll_x: False
            do_scroll_y: True
           
            MDBoxLayout:
                id: inventory_container
                orientation: 'vertical'
                spacing: "12dp"
                adaptive_height: True

<RecipeScreen>:
    name: "recipe_screen"
   
    MDBoxLayout:
        orientation: 'vertical'
        padding: "24dp"
        spacing: "20dp"
       
        MDBoxLayout:
            orientation: 'horizontal'
            size_hint_y: None
            height: "50dp"
            spacing: "12dp"
           
            MDRaisedButton:
                text: "← Back"
                on_release: root.manager.current = "home_screen"
               
            MDLabel:
                text: "Custom Inventory Recipes"
                font_style: "H6"
                halign: "center"
                theme_text_color: "Custom"
                text_color: 46/255, 125/255, 50/255, 1
               
            MDRaisedButton:
                text: "Reload Recipes"
                md_bg_color: 25/255, 118/255, 210/255, 1
                on_release: app.reload_recipes()

        MDBoxLayout:
            id: recipe_buttons_container
            orientation: 'vertical'
            spacing: "16dp"
            pos_hint: {"center_x": .5, "center_y": .6}
            adaptive_height: True
            width: "280dp"
            size_hint_x: None
'''

class HomeScreen(MDScreen):
    pass

class ScannerScreen(MDScreen):
    pass

class DashboardScreen(MDScreen):
    pass

class RecipeScreen(MDScreen):
    pass

class DoubleClickableLabel(ButtonBehavior, MDLabel):
    def __init__(self, item_index=None, on_double_click_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.item_index = item_index
        self.on_double_click_callback = on_double_click_callback

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if touch.is_double_tap and self.on_double_click_callback:
                self.on_double_click_callback(self.item_index)
                return True
        return super().on_touch_down(touch)


class NutriScanApp(MDApp):
    def build(self):
        self.title = "NutriScan AI"
        self.theme_cls.primary_palette = "Green"
        self.capture = None
        self.dialog = None
        self.last_scanned_code = None
        self.scanned_items_db = []  
        self.frame_count = 0
        self.failed_frame_count = 0
       
        self.current_scan_nutrients = {}
        self.manual_entry_name = ""
        self.active_recipes_map = {}

        # --- AI MODEL INITIALIZATION ---
        # Load the pre-trained MobileNetV2 model
        print("Loading AI Model...")
        self.ai_model = MobileNetV2(weights='imagenet')
        print("AI Model Ready.")

        return Builder.load_string(KV)

    def on_start(self):
        try:
            from pyzbar.pyzbar import decode
            self.barcode_decoder = decode
        except ImportError:
            print("Barcode system library (pyzbar) missing.")
            self.barcode_decoder = None

    # --- AI RECOGNITION LOGIC ---
    def run_ai_recognition(self):
        if not self.capture or not self.capture.isOpened():
            return

        ret, frame = self.capture.read()
        if ret:
            # 1. Image Preprocessing (MobileNetV2 expects 224x224)
            img = cv2.resize(frame, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # 2. Prediction
            preds = self.ai_model.predict(x)
            results = decode_predictions(preds, top=1)[0]
           
            # Format: (id, label, probability)
            label = results[0][1].replace('_', ' ').title()
            confidence = results[0][2]

            if confidence > 0.25:  # Confidence threshold
                self.manual_entry_name = label
                # Set up nutrient structure for manually identified AI item
                self.current_scan_nutrients = {
                    "name": label,
                    "grade": "AI Identified",
                    "calories": "N/A",
                    "proteins": "N/A",
                    "sugars": "N/A"
                }
               
                # Show result to user
                self.root.get_screen('scanner_screen').ids.status_label.text = f"AI Match: {label} ({confidence:.0%})"
                self.open_date_picker()
            else:
                self.root.get_screen('scanner_screen').ids.status_label.text = "AI unsure. Get closer to the food."

    def reload_recipes(self):
        try:
            screen = self.root.get_screen('recipe_screen')
            container = screen.ids.recipe_buttons_container
            container.clear_widgets()
            self.active_recipes_map.clear()

            scanned_ingredients = [item["name"] for item in self.scanned_items_db if item.get("name")]
            fallbacks = ["Water", "Salt", "Olive Oil", "Black Pepper", "Garden Herbs"]
            while len(scanned_ingredients) < 3:
                needed = 3 - len(scanned_ingredients)
                scanned_ingredients.extend(random.sample(fallbacks, min(needed, len(fallbacks))))

            colors = [[139/255, 195/255, 74/255, 1], [0/255, 150/255, 136/255, 1], [255/255, 112/255, 67/255, 1]]

            for i in range(3):
                sample_size = min(random.randint(1, 3), len(scanned_ingredients))
                chosen_ingredients = random.sample(scanned_ingredients, sample_size)
               
                dish_title = f"Medley: {' + '.join(chosen_ingredients[:2])}"
                if len(chosen_ingredients) > 2:
                    dish_title += " & Co."
               
                recipe_body = f"Ingredients Provided:\\n" + "\\n".join([f"• {ing}" for ing in chosen_ingredients])
                recipe_body += f"\\n\\nInstructions:\\n1. Clean and prepare {chosen_ingredients[0]}.\\n2. Combine ingredients and serve."
                recipe_body = recipe_body.replace('\\n', '\n')

                self.active_recipes_map[dish_title] = recipe_body

                btn = MDRaisedButton(
                    text=dish_title[:30] + '...' if len(dish_title) > 32 else dish_title,
                    size_hint_x=1,
                    md_bg_color=colors[i],
                    on_release=lambda x, title=dish_title: self.show_recipe(title)
                )
                container.add_widget(btn)
        except Exception as e:
            print(f"Recipe Error: {e}")

    def start_camera(self):
        if not self.capture:
            self.capture = cv2.VideoCapture(0)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
           
        self.last_scanned_code = None
        self.root.get_screen('scanner_screen').ids.status_label.text = "Align barcode or tap 'AI Identify'"
        Clock.schedule_interval(self.load_video, 1.0 / 30.0)

    def stop_camera(self):
        Clock.unschedule(self.load_video)
        if self.capture:
            self.capture.release()
            self.capture = None

    def load_video(self, *args):
        if not self.capture or not self.capture.isOpened():
            return
        ret, frame = self.capture.read()
        if not ret or frame is None:
            return

        self.frame_count += 1
        if self.barcode_decoder and self.frame_count % 10 == 0:
            detected_barcodes = self.barcode_decoder(frame)
            for barcode in detected_barcodes:
                barcode_data = barcode.data.decode('utf-8')
                if barcode_data != self.last_scanned_code:
                    self.last_scanned_code = barcode_data
                    Clock.schedule_once(lambda dt: self.process_barcode(barcode_data), 0)
                    break

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            corrected_frame = cv2.flip(rgb_frame, 0)
            height, width, _ = corrected_frame.shape
            texture = Texture.create(size=(width, height), colorfmt='rgb')
            texture.blit_buffer(corrected_frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            self.root.get_screen('scanner_screen').ids.camera_preview.texture = texture
        except:
            pass

    def open_manual_entry_flow(self):
        if self.dialog: self.dialog.dismiss()
        content_layout = MDBoxLayout(orientation="vertical", spacing="12dp", size_hint_y=None, height="60dp")
        self.manual_name_input = MDTextField(hint_text="Enter Item Description Name")
        content_layout.add_widget(self.manual_name_input)
        self.dialog = MDDialog(
            title="Manual Entry",
            type="custom",
            content_cls=content_layout,
            buttons=[
                MDFlatButton(text="Cancel", on_release=lambda x: self.dialog.dismiss()),
                MDRaisedButton(text="Set Expiry", on_release=self.process_manual_name_step)
            ],
        )
        self.dialog.open()

    def process_manual_name_step(self, *args):
        if not self.manual_name_input.text.strip(): return
        self.manual_entry_name = self.manual_name_input.text.strip()
        self.current_scan_nutrients = {"name": self.manual_entry_name, "grade": "N/A", "calories": "N/A", "proteins": "N/A", "sugars": "N/A"}
        self.dialog.dismiss()
        self.open_date_picker()

    def process_barcode(self, barcode):
        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
        try:
            response = requests.get(url, timeout=5).json()
            if response.get("status") == 1:
                product = response["product"]
                name = product.get("product_name", "Unknown Food")
                nutrients = product.get("nutriments", {})
                self.current_scan_nutrients = {
                    "name": name,
                    "grade": product.get("nutriscore_grade", "N/A").upper(),
                    "calories": f"{nutrients.get('energy-kcal_100g', 'N/A')} kcal",
                    "proteins": f"{nutrients.get('proteins_100g', 'N/A')}g",
                    "sugars": f"{nutrients.get('sugars_100g', 'N/A')}g"
                }
                self.show_product_dialog(name, f"Grade: {self.current_scan_nutrients['grade']}", active_product=True)
            else:
                self.show_product_dialog("Unknown Item", "Barcode not found.", fallback_allowed=True)
        except:
            self.show_product_dialog("Error", "Network connection failed.")

    def show_product_dialog(self, title, text, active_product=False, fallback_allowed=False):
        buttons = []
        if active_product:
            buttons.append(MDRaisedButton(text="Log Expiry", on_release=self.open_date_picker))
        elif fallback_allowed:
            buttons.append(MDRaisedButton(text="Try Manual", on_release=lambda x: self.open_manual_entry_flow()))
        buttons.append(MDFlatButton(text="Close", on_release=lambda x: self.dismiss_product_dialog()))
        self.dialog = MDDialog(title=title, text=text, buttons=buttons)
        self.dialog.open()

    def open_date_picker(self, *args):
        if self.dialog: self.dialog.dismiss()
        date_dialog = MDDatePicker()
        date_dialog.bind(on_save=self.save_item_expiry, on_cancel=lambda i, v: self.reset_scanner())
        date_dialog.open()

    def save_item_expiry(self, instance, value, date_range):
        self.scanned_items_db.append({
            "name": self.manual_entry_name or self.current_scan_nutrients.get("name"),
            "expiry_date": value.strftime('%Y-%m-%d'),
            "nutrients": self.current_scan_nutrients.copy()
        })
        self.manual_entry_name = "" # Reset
        instance.dismiss()
        self.reset_scanner()

    def reset_scanner(self, *args):
        self.last_scanned_code = None
        self.root.get_screen('scanner_screen').ids.status_label.text = "Ready for next item"

    def dismiss_product_dialog(self):
        if self.dialog: self.dialog.dismiss()
        self.reset_scanner()

    def view_dashboard(self):
        screen = self.root.get_screen('dashboard_screen')
        container = screen.ids.inventory_container
        container.clear_widgets()
       
        if not self.scanned_items_db:
            container.add_widget(MDLabel(text="Inventory is empty.", halign="center", height="100dp", size_hint_y=None))
        else:
            today = datetime.now().date()
            for idx, item in enumerate(self.scanned_items_db):
                expiry_dt = datetime.strptime(item["expiry_date"], "%Y-%m-%d").date()
                days_left = (expiry_dt - today).days
                card = MDCard(orientation='vertical', padding="16dp", size_hint_y=None, height="120dp", elevation=1)
               
                title = DoubleClickableLabel(text=item["name"], font_style="H6", bold=True, item_index=idx, on_double_click_callback=self.prompt_rename_dialog)
                status = MDLabel(text=f"Expires in {days_left} days", halign="right", theme_text_color="Hint")
               
                card.add_widget(title)
                card.add_widget(status)
                card.add_widget(MDFlatButton(text="Nutrients", on_release=lambda x, i=idx: self.show_isolated_nutrients(i)))
                container.add_widget(card)
               
        self.root.current = "dashboard_screen"

    def prompt_rename_dialog(self, item_index):
        item = self.scanned_items_db[item_index]
        content_layout = MDBoxLayout(orientation="vertical", spacing="12dp", size_hint_y=None, height="60dp")
        self.rename_input = MDTextField(text=item["name"])
        content_layout.add_widget(self.rename_input)
        self.dialog = MDDialog(title="Rename", type="custom", content_cls=content_layout,
            buttons=[MDFlatButton(text="Cancel", on_release=lambda x: self.dialog.dismiss()),
                     MDRaisedButton(text="Save", on_release=lambda x: self.execute_rename(item_index))])
        self.dialog.open()

    def execute_rename(self, item_index):
        self.scanned_items_db[item_index]["name"] = self.rename_input.text
        self.dialog.dismiss()
        self.view_dashboard()

    def show_isolated_nutrients(self, idx):
        item = self.scanned_items_db[idx]
        n = item["nutrients"]
        details = f"Grade: {n['grade']}\nCalories: {n['calories']}\nProteins: {n['proteins']}\nSugars: {n['sugars']}"
        self.dialog = MDDialog(title=item["name"], text=details, buttons=[MDRaisedButton(text="OK", on_release=lambda x: self.dialog.dismiss())])
        self.dialog.open()

    def show_recipe(self, dish_name):
        text = self.active_recipes_map.get(dish_name, "Recipe missing.")
        self.dialog = MDDialog(title=dish_name, text=text, buttons=[MDRaisedButton(text="Close", on_release=lambda x: self.dialog.dismiss())])
        self.dialog.open()

    def on_stop(self):
        self.stop_camera()

if __name__ == '__main__':
    NutriScanApp().run()
