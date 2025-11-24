from cssselect import HTMLTranslator

translator = HTMLTranslator()

def cquery_to_xpath(query_selector: str) -> str:
    """Compiles a query selector or a custom 

    Args:
        query_selector (str): Either a normal query selector, or a string in the shape of "{XML-ATTRIBUTE}§{QUERY-SELECTOR}"

    Returns:
        str: The resulting translated XPath
    """
    if "§" in query_selector:
        attribute, selector = query_selector.split("§")
        return f"{translator.css_to_xpath(selector)}/{attribute}"
    return translator.css_to_xpath(selector)

print(cquery_to_xpath("@value§foo.bar#baz > ban"))
cquery_to_xpath()