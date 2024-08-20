# from shiny.express import ui

# with ui.navset_pill(id="tab"):  
#     with ui.nav_panel("A"):
#         "Panel A content"

#     with ui.nav_panel("B"):
#         "Panel B content"

#     with ui.nav_panel("C"):
#         "Panel C content"

#     with ui.nav_menu("Other links"):
#         with ui.nav_panel("D"):
#             "Page D content"

#         "----"
#         "Description:"
#         with ui.nav_control():
#             # _blank: new tab
#             ui.a("Shiny", href="https://shiny.posit.co", target="_blank")

#==============================================

# from shiny.express import ui
# from shiny.express import input, render, ui

# ui.page_opts(fillable=False)

# # 열 2개(카드 2개 들어가 있는 상태)
# with ui.layout_columns():  
#     with ui.card():  
#         ui.card_header("Card 1 header")
#         ui.p("Card 1 body")
#         ui.input_slider("slider", "Slider", 0, 10, 5)

#     with ui.card():  
#         ui.card_header("Card 2 header")
#         ui.p("Card 2 body")
#         ui.input_text("text", "Add text", "")

# @render.text
# def text_out():
#     return f"Input value: {input.text()}"

#======================================

# from shiny.express import ui

# ui.page_opts(title="Page title")

# with ui.sidebar():
#     "Sidebar content"

# "Main content"

#=========================================

# from shiny.express import input, render, ui

# ui.page_opts(title="Page title")

# with ui.sidebar():
#      ui.input_selectize(
#     "var", "변수를 선택하세요",
#     choices=["bill_length_mm", "body_mass_g", "bill_depth_mm"] # 리스트
# )

# @render.plot
# def hist():
#     from matplotlib import pyplot as plt
#     from palmerpenguins import load_penguins

#     df = load_penguins()
#     # input.var(): string
#     df[input.var()].hist(grid=False) # input에서 선택한 var
#     plt.xlabel(input.var()) # x축 레이블
#     plt.ylabel("count")

#=====================================

# from shiny.express import input, render, ui

# ui.page_opts(title="Page title")

# with ui.sidebar():
#       ui.input_selectize(
#      "var", "변수를 선택하세요",
#      choices=["bill_length_mm", "body_mass_g", "bill_depth_mm"] # 리스트
# )

# with ui.nav_panel("Page 1"):
#     @render.plot
#     def hist():
#         from matplotlib import pyplot as plt
#         from palmerpenguins import load_penguins

#         df = load_penguins()
#         # input.var(): string
#         df[input.var()].hist(grid=False) # input에서 선택한 var
#         plt.xlabel(input.var()) # x축 레이블
#         plt.ylabel("count")

# with ui.nav_panel("Page 2"):
#     "Page 2 content"

#==========================================

from shiny.express import input, render, ui

ui.page_opts(title="팔머펭귄 부리 깊이 예측하기!")

with ui.sidebar():
    ui.input_selectize(
        "var", "펭귄 종을 선택해주세요!",
        choices=["Adelie", "Gentoo", "Chinstrap"]
    )
    ui.input_slider("slider1", "부리길이를 입력해주세요!", min=0, max=100, value=50)


@render.plot # 데코레이터
def scatter():
    from matplotlib import pyplot as plt
    import seaborn as sns
    from palmerpenguins import load_penguins
    import pandas as pd

    df = load_penguins()

    # !pip install scikit-learn
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    penguins=df.dropna()

    penguins_dummies = pd.get_dummies(
        penguins, 
        columns=['species'],
        drop_first=True
        )

    # x와 y 설정
    x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
    y = penguins_dummies["bill_depth_mm"]

    # 모델 학습
    model.fit(x, y)

    model.coef_
    model.intercept_

    regline_y=model.predict(x)

    import numpy as np
    index_1=np.where(penguins['species'] == "Adelie")
    index_2=np.where(penguins['species'] == "Gentoo")
    index_3=np.where(penguins['species'] == "Chinstrap")

    sns.scatterplot(data=df, 
                    x="bill_length_mm", 
                    y="bill_depth_mm",
                    hue="species")
    plt.plot(penguins["bill_length_mm"].iloc[index_1], regline_y[index_1], color="black")
    plt.plot(penguins["bill_length_mm"].iloc[index_2], regline_y[index_2], color="black")
    plt.plot(penguins["bill_length_mm"].iloc[index_3], regline_y[index_3], color="black")
    plt.xlabel("bill_length_mm")
    plt.ylabel("bill_depth_mm")
