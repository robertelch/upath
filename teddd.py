from textual.app import App, ComposeResult
from textual.widgets import Static, Digits
from textual.containers import Horizontal, VerticalScroll
from textual.reactive import reactive


VISIBLE_COUNT = 7
CENTER_INDEX = 3  # 3


class Row(Horizontal):
    def __init__(self, index: int, text: str, selected: bool = False):
        super().__init__(classes="row selected" if selected else "row")
        self.index = {
            0:1,
            1:2,
            2:3,
            3:0,
            4:4,
            5:5,
            6:6
        }[index]
        self.text = text

    def compose(self) -> ComposeResult:
        yield Digits(str(self.index), classes="digits")
        yield Static(self.text, classes="textbox")


class ScrollingList(App):

    BINDINGS = [
        ("1", "pick(1)"),
        ("2", "pick(2)"),
        ("3", "pick(3)"),
        ("4", "pick(4)"),
        ("5", "pick(5)"),
        ("6", "pick(6)"),
    ]

    CSS = """
    Screen {
        align: center middle;
    }

    VerticalScroll {
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
        self.displayed = slice(0,7)

    def compose(self) -> ComposeResult:
        self.scroller = VerticalScroll()
        yield self.scroller

    def on_mount(self) -> None:
        self.update_rows()
        self.scroll_center()

    # ----------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------

    def update_rows(self):
        self.log("Updating")
        
        for child in list(self.scroller.children):
            child.remove()

        for i, item in enumerate(self.items[self.displayed]):
            self.scroller.mount(
                Row(i, item, selected=(i == CENTER_INDEX))
            )
            self.log()

    def scroll_center(self):
        row_height = 3
        target_y = (self.selected * row_height) - (CENTER_INDEX * row_height)
        self.scroller.scroll_to(y=target_y, animate=True, duration=0.2)

    # ----------------------------------------------------------
    # Key handling via Actions (always works)
    # ----------------------------------------------------------

    def action_pick(self, n: int):
        """Called when pressing keys 1–6."""
        self.log("pressed sth")

        change = {
            1: -3,
            2: -2,
            3: -1,
            4: 1,
            5: 2,
            6: 3
        }[n]

        self.displayed = slice(self.displayed.start+change, self.displayed.stop+change)
        
        self.update_rows()
        self.scroll_center()


if __name__ == "__main__":
    ScrollingList().run()
