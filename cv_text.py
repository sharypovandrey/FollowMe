def add_text(text_type):
    if text_type == "left":
        bottomLeftCornerOfText = (10,500)
        fontColor = (255,255,255)
        text = "Turn left"
    elif text_type == "right":
        bottomLeftCornerOfText = (10,500)
        fontColor = (255,255,255)
        text = "Turn left"
    elif text_type == "forward":
        bottomLeftCornerOfText = (10,500)
        fontColor = (255,255,255)
        text = "Turn left"
    else:  # backward
        bottomLeftCornerOfText = (10,500)
        fontColor = (255,255,255)
        text = "Turn left"
    return text, \
        bottomLeftCornerOfText, \
        fontColor
