Viewed eval_metrics_20260517_112134.txt:1-9

I calculate the accuracy using two different strictness levels, which is handled directly inside our evaluation script (`scripts/eval_commonvoice.py`). 

Here is exactly how the math works under the hood:

### 1. Gender Accuracy (Strict)
The Mozilla dataset gives us labels like `male_masculine` and `female_feminine`. 
1. I strip these down to simply `male` and `female`.
2. I check if the model's prediction is an identical string match. 
3. **Calculation:** `(Total Exact Matches) / (Total Valid Samples)`

### 2. Age Bracket Accuracy (Exact)
Our model is constrained to output only one of 4 ordered brackets:
1. `18-30`
2. `31-45`
3. `46-60`
4. `60+`

I map the Mozilla text ages (e.g., "twenties" -> `18-30`, "sixties" -> `60+`). 
For the **Exact Accuracy**, the model's predicted bracket must be a 100% identical match to the true bracket. If a 45-year-old is predicted as falling in the `46-60` bracket, it counts as an absolute failure (0 points).

### 3. Age Bracket Accuracy (±1 Bracket Tolerance)
This is a more forgiving, real-world metric. Because a 30-year-old and a 31-year-old sound identical but fall into completely different brackets, this metric allows the model to be "off by one bracket" in either direction.

I treat the brackets as a numbered index (0 to 3). The prediction is considered **Correct** if the predicted index is `≤ 1` step away from the true index.

**Example: If the true age is `31-45` (Bracket #2)**
* Model predicts `31-45` (Bracket #2) ➡️ **Correct (Distance: 0)**
* Model predicts `18-30` (Bracket #1) ➡️ **Correct (Distance: 1)**
* Model predicts `46-60` (Bracket #3) ➡️ **Correct (Distance: 1)**
* Model predicts `60+` (Bracket #4) ➡️ **Incorrect (Distance: 2)**

This is why your ±1 Bracket Accuracy sits at nearly 95%, while the Exact Accuracy is hovering around 33%—the model is consistently getting the age *almost* perfectly right, but is getting penalized by the rigid boundaries of the exact buckets.