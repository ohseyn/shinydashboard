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

# from shiny.express import input, render, ui

# ui.page_opts(title="팔머펭귄 부리 깊이 예측하기!")

# with ui.sidebar():
#     ui.input_selectize(
#         "var", "펭귄 종을 선택해주세요!",
#         choices=["Adelie", "Gentoo", "Chinstrap"]
#     )
#     ui.input_slider("slider1", "부리길이를 입력해주세요!", min=0, max=100, value=50)

#     # @render.text
#     # def cal_depth():
#     #     y_hat = 0.2 * input.slider1() + 10.56
#     #     return f"부리길이 예상치: {round(y_hat, 3)}"

#     @render.text
#     def cal_depth():
#         if input.var() == "Adelie":
#             y_hat = 0.2 * input.slider1() + 10.56
#         elif input.var() == "Chinstrap":
#             y_hat = 0.2 * input.slider1() + (-1.93) + 10.56
#         else:
#             y_hat = 0.2 * input.slider1() + (-5.1) + 10.56
#         return f"부리길이 예상치: {round(y_hat, 3)}"


# @render.plot # 데코레이터
# def scatter():
#     from matplotlib import pyplot as plt
#     import seaborn as sns
#     from palmerpenguins import load_penguins
#     import pandas as pd

#     df = load_penguins()

#     # !pip install scikit-learn
#     from sklearn.linear_model import LinearRegression

#     model = LinearRegression()
#     penguins=df.dropna()

#     penguins_dummies = pd.get_dummies(
#         penguins, 
#         columns=['species'],
#         drop_first=True
#         )

#     # x와 y 설정
#     x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
#     y = penguins_dummies["bill_depth_mm"]

#     # 모델 학습
#     model.fit(x, y)

#     model.coef_
#     model.intercept_

#     x=pd.DataFrame({
#            "bill_length_mm": 15,
#            "species_Chinstrap": [False],
#            "species_Gentoo": [False]
#         })
#     regline_y=model.predict(x)

#     regline_y=model.predict(x)
#     # regline_y=model.predict([[15, False, False]])

#     import numpy as np
#     index_1=np.where(penguins['species'] == "Adelie")
#     index_2=np.where(penguins['species'] == "Gentoo")
#     index_3=np.where(penguins['species'] == "Chinstrap")

#     sns.scatterplot(data=df, 
#                     x="bill_length_mm", 
#                     y="bill_depth_mm",
#                     hue="species")
#     plt.plot(penguins["bill_length_mm"].iloc[index_1], regline_y[index_1], color="black")
#     plt.plot(penguins["bill_length_mm"].iloc[index_2], regline_y[index_2], color="black")
#     plt.plot(penguins["bill_length_mm"].iloc[index_3], regline_y[index_3], color="black")
#     plt.xlabel("bill_length_mm")
#     plt.ylabel("bill_depth_mm")

#============================================

# from shiny.express import input, render, ui

# ui.page_opts(title="팔머펭귄 부리 깊이 예측하기!")

# with ui.sidebar():
#     ui.input_selectize(
#         "var", "펭귄 종을 선택해주세요!",
#         choices=["Adelie", "Gentoo", "Chinstrap"]
#     )
#     ui.input_slider("slider1", "부리길이를 입력해주세요!", min=0, max=100, value=50)

#     @render.text
#     def cal_depth():
#         from palmerpenguins import load_penguins
#         from sklearn.linear_model import LinearRegression
#         import pandas as pd
#         import numpy as np
#         penguins = load_penguins().dropna()
#         penguins_dummies = pd.get_dummies(penguins, 
#                                 columns=['species'],
#                                 drop_first=False)
#         x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
#         y = penguins_dummies['bill_depth_mm']

#         model=LinearRegression()
#         model.fit(x, y)
#         # 카테고리
#         input_df = pd.DataFrame({
#             # 숫자
#             "bill_length_mm" : [input.slider1()],
#             # 종 알려줌
#             "species": pd.Categorical([input.var()],
#                                       categories=["Adelie", "Chinstrap", "Gentoo"])
            
#         })
#         # dummies 처리
#         # 모델이 사용할 수 있는 형태로 바꾸기 위해 
#         # 범주형 변수를 이진 변수(True, False)로 변환
#         input_df = pd.get_dummies(
#             input_df, 
#             columns = ["species"],
#             drop_first = True
#         )
#         y_hat = model.predict(input_df)
#         y_hat = float(y_hat)
#         return f'부리 깊이 예상치: {y_hat:.2f}'

# @render.plot # 데코레이터
# def scatter():
#     from palmerpenguins import load_penguins
#     import seaborn as sns
#     from matplotlib import pyplot as plt
#     from sklearn.linear_model import LinearRegression
#     import pandas as pd
#     import numpy as np

#     df = load_penguins()

#     model = LinearRegression()
#     penguins=df.dropna()

#     penguins_dummies = pd.get_dummies(
#         penguins, 
#         columns=['species'],
#         drop_first=True
#         )

#     # x와 y 설정
#     x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
#     y = penguins_dummies["bill_depth_mm"]

#     # 모델 학습
#     model.fit(x, y)

#     model.coef_
#     model.intercept_

#     regline_y=model.predict(x)

#     index_1=np.where(penguins['species'] == "Adelie")
#     index_2=np.where(penguins['species'] == "Gentoo")
#     index_3=np.where(penguins['species'] == "Chinstrap")

#     sns.scatterplot(data=df, 
#                     x="bill_length_mm", 
#                     y="bill_depth_mm",
#                     hue="species")
#     plt.plot(penguins["bill_length_mm"].iloc[index_1], regline_y[index_1], color="black")
#     plt.plot(penguins["bill_length_mm"].iloc[index_2], regline_y[index_2], color="black")
#     plt.plot(penguins["bill_length_mm"].iloc[index_3], regline_y[index_3], color="black")
#     plt.xlabel("bill_length_mm")
#     plt.ylabel("bill_depth_mm")

#===========================================

# from shiny.express import input, render, ui

# ui.page_opts(title="팔머펭귄 부리 깊이 예측하기!")

# with ui.sidebar():
#     ui.input_selectize(
#         "var", "펭귄 종을 선택해주세요!",
#         choices=["Adelie", "Gentoo", "Chinstrap"]
#     )
#     ui.input_slider("slider1", "부리길이를 입력해주세요!", min=0, max=100, value=50)

#     @render.text
#     def cal_depth():
#         from palmerpenguins import load_penguins
#         from sklearn.linear_model import LinearRegression
#         import pandas as pd
#         import numpy as np
#         penguins = load_penguins().dropna()
#         penguins_dummies = pd.get_dummies(penguins, 
#                                 columns=['species'],
#                                 drop_first=False)
#         x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
#         y = penguins_dummies['bill_depth_mm']

#         model=LinearRegression()
#         model.fit(x, y)
#         # 카테고리
#         input_df = pd.DataFrame({
#             # 숫자
#             "bill_length_mm" : [input.slider1()],
#             # 종 알려줌
#             "species": pd.Categorical([input.var()],
#                                       categories=["Adelie", "Chinstrap", "Gentoo"])
            
#         })
#         # dummies 처리
#         # 모델이 사용할 수 있는 형태로 바꾸기 위해 
#         # 범주형 변수를 이진 변수(True, False)로 변환
#         input_df = pd.get_dummies(
#             input_df, 
#             columns = ["species"],
#             drop_first = True
#         )
#         y_hat = model.predict(input_df)
#         y_hat = float(y_hat)
#         return f'부리 깊이 예상치: {y_hat:.2f}'

# @render.plot # 데코레이터
# def scatter():
#     from palmerpenguins import load_penguins
#     import seaborn as sns
#     from matplotlib import pyplot as plt
#     from sklearn.linear_model import LinearRegression
#     import pandas as pd
#     import numpy as np

#     df = load_penguins()

#     model = LinearRegression()
#     penguins=df.dropna()

#     penguins_dummies = pd.get_dummies(
#         penguins, 
#         columns=['species'],
#         drop_first=True
#         )

#     # x와 y 설정
#     x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
#     y = penguins_dummies["bill_depth_mm"]

#     # 여기가 추가됨
#     x["bill_Chinstrap"] = x["bill_length_mm"] * x["species_Chinstrap"]
#     x["bill_Gentoo"] = x["bill_length_mm"] * x["species_Gentoo"]
    
#     # 모델 학습
#     model.fit(x, y)

#     model.coef_
#     model.intercept_

#     regline_y=model.predict(x)

#     index_1=np.where(penguins['species'] == "Adelie")
#     index_2=np.where(penguins['species'] == "Gentoo")
#     index_3=np.where(penguins['species'] == "Chinstrap")

#     sns.scatterplot(data=df, 
#                     x="bill_length_mm", 
#                     y="bill_depth_mm",
#                     hue="species")
#     plt.plot(penguins["bill_length_mm"].iloc[index_1], regline_y[index_1], color="black")
#     plt.plot(penguins["bill_length_mm"].iloc[index_2], regline_y[index_2], color="black")
#     plt.plot(penguins["bill_length_mm"].iloc[index_3], regline_y[index_3], color="black")
#     plt.xlabel("bill_length_mm")
#     plt.ylabel("bill_depth_mm")

#==================================

# from palmerpenguins import load_penguins
# import seaborn as sns
# from matplotlib import pyplot as plt
# from sklearn.linear_model import LinearRegression
# import pandas as pd
# import numpy as np
# # !pip install patsy
# import patsy

# df = load_penguins()
# penguins=df.dropna()
# model = LinearRegression()

# # patsy을 사용하여 수식(회귀 모델)으로 상호작용 항 생성
# # 0 + (- 1) 는 절편 제거
# # β0(Intercept/y절편)은 다 더하니까 다 입력값 1(True) 나옴)
# # 종속변수 ~ 독립변수1 + 독립변수2
# formula = "bill_depth_mm ~ 0 + bill_length_mm * species"
# # formula = "bill_depth_mm ~ 0 + bill_length_mm + body_mass_g + flipper_length_mm + species"
# #formula = "bill_depth_mm ~ 0 + bill_length_mm * body_mass_g * flipper_length_mm * species"
# # 설계 매트릭스(독립 변수(특징)의 값 x)와 종속 변수 벡터 y를 생성
# y, x = patsy.dmatrices(formula, penguins, return_type="dataframe")
# x=x.iloc[:,1:]

# # 모델 학습
# model.fit(x, y)

# model.coef_
# model.intercept_

#========================================

# from matplotlib import pyplot as plt
# import seaborn as sns
# from palmerpenguins import load_penguins
# import pandas as pd

# df = load_penguins()

# # !pip install scikit-learn
# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# penguins=df.dropna()

# penguins_dummies = pd.get_dummies(
#     penguins, 
#     columns=['species'],
#     drop_first=True
#     )

# # x와 y 설정
# x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
# y = penguins_dummies["bill_depth_mm"]

# # 모델 학습
# model.fit(x, y)

# model.coef_
# model.intercept_

# regline_y=model.predict(x)

# x=pd.DataFrame({
#     'bill_length_mm': 15.0,
#     'species': pd.Categorical(['Adelie'], 
#                                 categories=['Adelie', 'Chinstrap', 'Gentoo'])
#     })
# x = pd.get_dummies(
#     x, 
#     columns=['species'],
#     drop_first=True
#     )
# "y_hat"=model.predict(x)
# y_hat
