isa={ "bird":"animal",
     "dog":"animal",
     "sparrow":"bird"}

can_do={"dog":"bark",
        "bird":"fly"}

has_a={"dog":"tail",
       "bird":"wing",
       "animal":"cells"}

def get_superclass(concept):
    if concept in isa:
        return isa[concept]
    return None


def check_relation(concept, relation, value):
    current = concept

    while current:
        if relation == "can":
            if current in can_do and can_do[current] == value:
                return True

        elif relation == "has":
            if current in has_a and has_a[current] == value:
                return True

        elif relation == "isa":
            if current == value:
                return True

        current = get_superclass(current)

    return False


# Preset Queries
queries = [
    ("sparrow","isa","animal"),
    ("dog","can","fly"),
    ("dog","can","bark"),
    ("sparrow","can","fly"),
    ("animal","has","cells")
]

print("Semantic Network Query Results:\n")

for concept, relation, value in queries:
    result = check_relation(concept, relation, value)
    print(concept, relation, value, "->", result)