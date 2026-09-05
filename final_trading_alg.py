"""
Highly sophisticated PnL generator. 
`predicted_trend` will be given by pedestal model predictions
Up? Buy. Down? Sell. Unsure? Do nothing.
"""

making_money = True

predicted_trend = ... # For now.
def buy(amount):
    return ... # For now.
def sell(amount):
    return ... # For now.

while making_money:
    if predicted_trend == "I think its going up":
        buy("lots")
    elif predicted_trend == "I think its going down":
        sell("what we have")
    elif predicted_trend == "Im not sure..":
        pass