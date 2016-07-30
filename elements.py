import parsel
import six


def to_text(s, encoding='utf-8'):
    if isinstance(s, six.text_type):
        return s
    return s.decode(encoding)


def get_parent(selector):
    return parsel.Selector(root=selector.root.getparent())


def get_all_leaves(page):
    sel = parsel.Selector(text=to_text(page))
    return sel.xpath("//*[not(*)][./text()]")


def get_all_meta_content(page):
    sel = parsel.Selector(text=to_text(page))
    return sel.xpath("//meta[@content]")
