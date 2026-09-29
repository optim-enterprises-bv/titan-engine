#!/usr/bin/env python3
"""Build q100.json: [question, answer regex] x 100 -- 40 arithmetic / word problems (GSM8K-style, answers computed
here), 30 facts, 30 Python output predictions (answers obtained by running the snippet here)."""
import contextlib, io, json, re

def num(x):  # regex for an integer answer, allowing thousands separators
    s = f"{x:,}"
    return r"(?<![\d.])(" + re.escape(str(x)) + "|" + re.escape(s) + r")(?![\d])"

Q = []
# ---- arithmetic / word problems
arith = [
    ("What is 48 * 27?", 48 * 27),
    ("What is 1234 + 5678?", 1234 + 5678),
    ("What is 9000 - 4567?", 9000 - 4567),
    ("What is 3 cubed plus 4 cubed?", 27 + 64),
    ("What is 144 divided by 12, multiplied by 7?", 144 // 12 * 7),
    ("What is 25% of 360?", 90),
    ("What is the sum of the integers from 1 to 50?", sum(range(1, 51))),
    ("What is 17 squared?", 289),
    ("What is 2 to the power of 16?", 2 ** 16),
    ("What is the greatest common divisor of 84 and 126?", 42),
    ("What is the least common multiple of 12 and 18?", 36),
    ("How many seconds are in 2 hours and 15 minutes?", 2 * 3600 + 15 * 60),
    ("What is 7! (7 factorial)?", 5040),
    ("What is 999 * 3?", 2997),
    ("What is 56 * 56 - 55 * 55?", 56 * 56 - 55 * 55),
    ("Tom has 3 boxes with 24 apples each. He gives away 19 apples. How many apples does he have left?", 3 * 24 - 19),
    ("A shirt costs $40 and is discounted by 15%. What is the sale price in dollars?", 34),
    ("A car travels at 65 km/h for 4 hours. How many kilometres does it travel?", 260),
    ("Sara reads 12 pages a day. How many days does she need to read a 228-page book?", 19),
    ("A recipe needs 3 eggs per cake. How many eggs are needed for 14 cakes?", 42),
    ("There are 5 classes of 28 students and 4 classes of 31 students. How many students are there in total?", 5 * 28 + 4 * 31),
    ("A worker earns $18 per hour and works 37 hours. How many dollars does she earn?", 18 * 37),
    ("A rectangle is 23 cm long and 17 cm wide. What is its area in square centimetres?", 23 * 17),
    ("A rectangle is 23 cm long and 17 cm wide. What is its perimeter in centimetres?", 2 * (23 + 17)),
    ("John buys 4 notebooks at $2.50 each and a pen for $1.75. He pays with a $20 bill. How much change does he get, in dollars?", "8.25"),
    ("A tank holds 450 litres and is filled at 18 litres per minute. How many minutes does it take to fill from empty?", 25),
    ("If 8 workers build a wall in 15 days, how many days would 12 workers take at the same rate?", 10),
    ("A bag has 6 red, 9 blue and 15 green marbles. What percentage of the marbles are green?", 50),
    ("Mia is 3 times as old as her son. The son is 12. How old will Mia be in 5 years?", 41),
    ("A train leaves at 09:40 and arrives at 13:05. How many minutes is the journey?", 205),
    ("The average of five numbers is 18. Four of them are 12, 20, 15 and 25. What is the fifth?", 90 - 72),
    ("A farmer has 120 chickens and sells 3/8 of them. How many chickens remain?", 75),
    ("A phone costs $600. The price rises by 10% and then falls by 10%. What is the final price in dollars?", 594),
    ("A pizza is cut into 12 slices. Ann eats 1/4 of it and Ben eats 1/3 of it. How many slices are left?", 5),
    ("What is 15% of 15% of 10000?", 225),
    ("How many prime numbers are there between 1 and 30?", 10),
    ("What is 1001 divided by 7?", 143),
    ("What is 0.125 written as a fraction in lowest terms? Give it as a/b.", "1/8"),
    ("What is the remainder when 1000 is divided by 7?", 1000 % 7),
    ("A store sells pencils in packs of 12. How many packs are needed for 150 pencils?", 13),
]
for q, a in arith:
    Q.append([q, num(a) if isinstance(a, int) else r"(?<![\d.])" + re.escape(a) + r"(?![\d])"])

# ---- facts
facts = [
    ("What is the capital of Canada?", r"ottawa"),
    ("What is the capital of Japan?", r"tokyo"),
    ("What is the capital of Kenya?", r"nairobi"),
    ("What is the chemical symbol for sodium?", r"\bNa\b"),
    ("What is the chemical symbol for iron?", r"\bFe\b"),
    ("How many chromosomes do humans normally have?", r"\b46\b"),
    ("Who painted the Mona Lisa?", r"leonardo|da vinci"),
    ("Who developed the theory of general relativity?", r"einstein"),
    ("What is the largest planet in the Solar System?", r"jupiter"),
    ("What is the hardest natural mineral?", r"diamond"),
    ("In which year did the first human land on the Moon?", r"\b1969\b"),
    ("What is the longest river in Africa?", r"\bnile\b"),
    ("Which element has atomic number 8?", r"oxygen"),
    ("What is the speed of light in vacuum, in kilometres per second, rounded to the nearest thousand?", r"300[, ]?000"),
    ("Who wrote 'Romeo and Juliet'?", r"shakespeare"),
    ("What is the currency of the United Kingdom?", r"pound|sterling"),
    ("How many continents are there?", r"\b(7|seven)\b"),
    ("What is the freezing point of water in degrees Fahrenheit?", r"\b32\b"),
    ("Which organ pumps blood through the human body?", r"heart"),
    ("What is the smallest planet in the Solar System?", r"mercury"),
    ("Who was the first President of the United States?", r"washington"),
    ("What language has the most native speakers in the world?", r"mandarin|chinese"),
    ("What is the tallest mountain on Earth above sea level?", r"everest"),
    ("Which programming language was created by Guido van Rossum?", r"python"),
    ("What does CPU stand for?", r"central processing unit"),
    ("How many bits are in a byte?", r"\b(8|eight)\b"),
    ("What is the main gas in Earth's atmosphere?", r"nitrogen"),
    ("In which country are the pyramids of Giza?", r"egypt"),
    ("What is the square root of 225?", r"\b15\b"),
    ("Which company developed the Windows operating system?", r"microsoft"),
]
Q += [list(f) for f in facts]

# ---- Python output prediction (answers from running the code)
code = [
    "print(len('hello world'))",
    "print(sum([3, 5, 7, 9]))",
    "print(10 // 3, 10 % 3)",
    "x = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x))",
    "print('abc' * 3)",
    "print(sorted([5, 2, 9, 1])[1])",
    "print(2 ** 3 ** 2)",
    "print([i * i for i in range(5)])",
    "print('Python'[1:4])",
    "d = {'a': 1, 'b': 2}\nd['c'] = d['a'] + d['b']\nprint(sum(d.values()))",
    "print(max('banana'))",
    "s = 0\nfor i in range(1, 11):\n    if i % 2 == 0:\n        s += i\nprint(s)",
    "print(int('101', 2))",
    "print(len(set([1, 2, 2, 3, 3, 3])))",
    "print('-'.join(['a', 'b', 'c']))",
    "print(round(7 / 2))",
    "print(list(range(10, 0, -3)))",
    "def f(n):\n    return 1 if n <= 1 else n * f(n - 1)\nprint(f(5))",
    "print('Hello'.lower().count('l'))",
    "a, b = 3, 5\na, b = b, a + b\nprint(a, b)",
    "print(bool([]), bool([0]))",
    "print(7 / 2)",
    "print('racecar'[::-1] == 'racecar')",
    "x = 5\nx *= 3\nx -= 4\nprint(x)",
    "print(abs(-12) + len([None, None]))",
    "print(' '.join(w.capitalize() for w in 'the quick fox'.split()))",
    "n = 0\nwhile n * n < 50:\n    n += 1\nprint(n)",
    "print(min([4, -2, 7], key=abs))",
    "print(hex(255))",
    "fib = [0, 1]\nfor _ in range(6):\n    fib.append(fib[-1] + fib[-2])\nprint(fib[-1])",
]
for c in code:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(c, {})
    out = buf.getvalue().strip()
    Q.append([f"What does this Python code print?\n```python\n{c}\n```", r"(?m)^\s*(?:output:\s*)?`*(?:text|python)?\s*['\"]?" + re.escape(out) + r"['\"]?`*\.?\s*$"])
assert len(Q) == 100, len(Q)
json.dump(Q, open("q100.json", "w"), indent=1)
print(len(Q), "questions")
