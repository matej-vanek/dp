from yattag import Doc, indent

doc, tag, text, line = Doc(
    defaults = {'ingredient': ['chocolate', 'coffee']}
).ttl()

with tag('form', action = ""):
    line('label', 'Select one or more ingredients')
    with doc.select(name = 'ingredient', multiple = "multiple"):
        for value, description in (
            ("chocolate", "Dark Chocolate"),
            ("almonds", "Roasted almonds"),
            ("honey", "Acacia honey"),
            ("coffee", "Ethiopian coffee")
        ):
            with doc.option(value = value):
                text(description)
    doc.stag('input', type = "submit", value = "Validate")

with open("/home/matejvanek/dp/Prace/dashboard2.html", "w") as f:
    f.write(indent(doc.getvalue()))
