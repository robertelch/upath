class Extractor: ...
class Placeholder: ...
class Placeholder:
    def __init__(self, _class: str) -> None:
        self._class = _class

    def replace(self, classes: dict[str, type[Extractor]]) -> Extractor:
        for _class in classes.values():
            for name, value in vars(_class).items():