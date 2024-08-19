from shiny.express import input, render, ui

ui.input_selectize(
    "var", "변수를 선택하세요",
    choices=["bill_length_mm", "body_mass_g", "bill_depth_mm"] # 리스트
)

@render.plot
def hist():
    from matplotlib import pyplot as plt
    from palmerpenguins import load_penguins

    df = load_penguins()
    # input.var(): string
    df[input.var()].hist(grid=False) # input에서 선택한 var
    plt.xlabel(input.var()) # x축 레이블
    plt.ylabel("count")