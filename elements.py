import parsel


def get_all_leaves(page):
    sel = parsel.Selector(text=page.decode('utf-8'))
    return sel.xpath("//*[not(*)][./text()]")


def get_all_meta_content(page):
    sel = parsel.Selector(text=page.decode('utf-8'))
    return sel.xpath("//meta[@content]")
