from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.card import MDCard
from kivymd.uix.screen import MDScreen
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.button import MDRoundFlatButton
from kivy.uix.anchorlayout import AnchorLayout
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from asr_module import infer_latest_command  # Import ASR function

# Set a default window size for desktop preview
Window.size = (360, 640)

class SmartHomeApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "BlueGray"

        self.screen = MDScreen()

        # Draw a background color
        with self.screen.canvas.before:
            Color(31/255, 36/255, 58/255, 1)  # Dark blue
            self.bg_rect = Rectangle(size=self.screen.size, pos=self.screen.pos)
        self.screen.bind(size=self._update_rect, pos=self._update_rect)

        # Scrollable layout to hold all content
        scroll = MDScrollView()
        self.layout = MDBoxLayout(orientation="vertical", padding=20, spacing=20, size_hint_y=None)
        self.layout.bind(minimum_height=self.layout.setter("height"))

        # Header greeting block
        greeting_box = MDBoxLayout(orientation="vertical", size_hint=(1, None), height=100, spacing=5)
        greeting_box.add_widget(MDLabel(
            text="Hello!",
            font_style="H5",
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1)
        ))
        greeting_box.add_widget(MDLabel(
            text="Welcome to Home",
            font_style="Caption",
            theme_text_color="Custom",
            text_color=(1, 1, 1, 0.8)
        ))
        self.layout.add_widget(greeting_box)

        # Grid of room cards
        self.grid = MDGridLayout(cols=2, spacing=20, adaptive_height=True)
        self.cards = []

        # Define available rooms and their card titles
        rooms = [
            ("Music", ""),
            ("Living Room", "")
        ]

        for name, devices in rooms:
            card = MDCard(
                size_hint=(1, None),
                height=120,
                radius=[20],
                md_bg_color=(47/255, 51/255, 75/255, 1),
                orientation='vertical',
                padding=0,
                line_width=2,
                line_color=[0, 0, 0, 0]
            )

            label = MDLabel(
                text=name,
                halign="center",
                valign="center",
                theme_text_color="Custom",
                text_color=(1, 1, 1, 1),
                font_style="Body1"
            )
            label.bind(size=label.setter("text_size"))
            card.add_widget(label)

            self.cards.append((card, label))
            self.grid.add_widget(card)

        self.layout.add_widget(self.grid)
        scroll.add_widget(self.layout)
        self.screen.add_widget(scroll)

        # Add "Speak" button at the bottom
        speak_button = MDRoundFlatButton(
            text="Speak",
            size_hint=(None, None),
            size=("120dp", "48dp"),
            pos_hint={"center_x": 0.5},
            md_bg_color=(1, 1, 1, 1),
            text_color=(0, 0, 0, 1),
            font_size="16sp"
        )
        speak_button.bind(on_release=self.on_speak_pressed)

        anchor = AnchorLayout(anchor_x="center", anchor_y="bottom", padding=[0, 20, 0, 20])
        anchor.add_widget(speak_button)
        self.screen.add_widget(anchor)

        return self.screen

    def highlight_card(self, obj, action):
        """Highlight the corresponding card based on prediction."""
        for c, l in self.cards:
            c.line_color = [0, 0, 0, 0]
            c.md_bg_color = (47 / 255, 51 / 255, 75 / 255, 1)
            l.text_color = (1, 1, 1, 1)

        # Map model prediction to card title
        name_map = {
            "music": "Music",
            "lights": "Living Room"
        }

        room_label = name_map.get(obj.lower())
        if not room_label:
            return

        for card, label in self.cards:
            if label.text == room_label:
                card.line_color = [1, 1, 1, 1]
                if action.lower() == "on":
                    card.md_bg_color = (60 / 255, 65 / 255, 100 / 255, 1)
                    label.text_color = (0, 1, 0, 1)  # Green for ON
                else:
                    card.md_bg_color = (30 / 255, 33 / 255, 50 / 255, 1)
                    label.text_color = (1, 0, 0, 1)  # Red for OFF
                break

    def on_speak_pressed(self, instance):
        """Callback when Speak button is pressed. Run ASR."""
        text, action, obj, truth = infer_latest_command()
        print("Recognized:", text)
        print("Action:", action)
        print("Object:", obj)
        print("Ground Truth:", truth)
        self.highlight_card(obj, action)

    def _update_rect(self, instance, value):
        self.bg_rect.size = instance.size
        self.bg_rect.pos = instance.pos


SmartHomeApp().run()
