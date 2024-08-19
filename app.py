from shiny.express import input, render, ui

# input
ui.input_slider("n", "숫자를 선택하세요", 0, 100, 20)

# output
@render.code
def txt():
    return f"n*2 is {input.n() * 2}"

# run shiny app 눌러서 실행
# ctrl+c 누르면 꺼짐