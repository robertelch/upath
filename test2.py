from textual.widget import Widget
from textual.app import ComposeResult, App
from textual.widgets import Static, Digits
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive


VISIBLE_COUNT = 7
CENTER_INDEX = 3  # 3


class Row(Horizontal):
    def __init__(self, index: int, text: str, selected: bool = False):
        super().__init__(classes="row selected" if selected else "row")
        self.index = {0:"1", 1:"2", 2:"3", 3:"0", 4:"4", 5:"5", 6:"6"}[index]
        self.text = text

    def compose(self) -> ComposeResult:
        yield Digits(self.index, classes="digits")
        yield Static(self.text, classes="textbox")


class ScrollingList(Widget, can_focus=True):

    BINDINGS = [
        ("1", "pick(1)"),
        ("2", "pick(2)"),
        ("3", "pick(3)"),
        ("4", "pick(4)"),
        ("5", "pick(5)"),
        ("6", "pick(6)"),
    ]

    DEFAULT_CSS = """
    Screen {
        align: center middle;
    }

    Vertical {
        width: 70;
        height: auto;
        border: solid #444;
    }

    .row {
        height: 3;
        width: 100%;
    }

    .digits {
        width: auto;
        height: 3;
        margin-right: 1;
        content-align: center middle;
    }

    .textbox {
        border: solid #777;
        height: 3;
        width: 100%;
        padding: 0 1;
        content-align: left middle;
    }

    .selected .textbox {
        border: heavy green;
    }
    """

    selected = reactive(3)

    def __init__(self):
        super().__init__()
        self.items = [f"Item number {i}" for i in range(1, 30)]
        self.displayed = slice(0, 7)

    def compose(self) -> ComposeResult:
        self.scroller = Vertical()
        yield self.scroller

    def on_mount(self) -> None:
        self.update_rows()

    # ----------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------

    def update_rows(self):
        self.log("Updating")

        for child in list(self.scroller.children):
            child.remove()
        self.log("Hi")
        for i, item in enumerate(self.items[self.displayed]):
            self.scroller.mount(
                Row(i, item, selected=(i == CENTER_INDEX))
            )
    # ----------------------------------------------------------
    # Key handling via Actions (always works)
    # ----------------------------------------------------------

    def action_pick(self, n: int):
        """Called when pressing keys 1–6."""
        self.log("pressed sth")
        change = {1:-3, 2:-2, 3:-1, 4:1, 5:2, 6:3}[n]

        self.displayed = slice(
            self.displayed.start + change,
            self.displayed.stop + change
        )

        self.update_rows()

# ----------------------------------------------------------
# Example: using the component inside an App
# ----------------------------------------------------------

class Demo(App):
    CSS = "ScrollingList { align: center middle; }"

    def compose(self):
        yield ScrollingList()


if __name__ == "__main__":
    Demo().run()
