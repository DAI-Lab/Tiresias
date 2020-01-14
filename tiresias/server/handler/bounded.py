def handle_bounded(query, data):
    values = []
    for row in data:
        values.extend(row)
    return values
