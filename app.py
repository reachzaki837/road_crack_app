You have two syntax errors in your code:

1. **Non-ASCII apostrophe in comment**  
   In this line:
   ```python
   if not os.path.isfile(log_file):  # don’t overwrite existing
   ```
   The apostrophe in `don’t` is a curly quote (Unicode), not a plain ASCII `'`.  
   **Fix:** Change `don’t` to `don't`.

2. **Misspelled print function**  
   At the end of your code:
   ```python
   prnt("Hello")
   ```
   This should be `print("Hello")`.

---

**Corrected code snippets:**

**1. Change:**
```python
if not os.path.isfile(log_file):  # don’t overwrite existing
```
**To:**
```python
if not os.path.isfile(log_file):  # don't overwrite existing
```

**2. Change:**
```python
prnt("Hello")
```
**To:**
```python
print("Hello")
```

---

**Summary of fixes:**

- Replace all curly quotes/apostrophes with straight ones.
- Fix the typo `prnt` to `print`.

---

**Your code will now run without syntax errors.**