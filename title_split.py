def list_split(l: list[str], symbol: str):
    return [s for i in l for a in i.split(symbol) if (s := a.strip)]

def split_titles(main_title: str, other_titles: str, splitting_symbols: list[str]):
    titles = [other_titles]
    for symbol in splitting_symbols:
        if not symbol in main_title and symbol in other_titles:
            titles = list_split(titles, symbol)
    return [main_title.strip(), *titles]
