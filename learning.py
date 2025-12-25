import streamlit as st

st.title("Suresh Web Framework")
st.header("Learning")
food_list = ["egg","rice","tomoto"]
food = st.text_input("Enter The Fav Food")

button =st.button("Check Food")

if button == True:
    if food.lower() in food_list:
        st.text("Your Food having the List")
    else:
        st.text("Your Food Not Having the List")
    # st.text(food)


mutli = st.multiselect("Select Your FAV",["Suresh","Harsha"],'Suresh')
# st.text(mutli)+


if mutli == ["Suresh"]:
    st.title(mutli)
else:
    st.title("Not Slelct")


# amount_slider = st.select_slider("amount",options=[400000])



