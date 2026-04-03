# Task 10: JSON Transactions

**Source:** https://highload.fun/tasks/10

## Description

You have a data stream of records sent to STDIN. Each record is encoded in JSON.

Try to find the total amount of external transactions (where `record.user_id != record.transaction.to_user_id`) in USD for the shortest time.

## Input

JSON records via STDIN with the following structure:

```json
{
  "user_id": 0,
  "currency": "",
  "transactions": [
    {
      "amount": 0,
      "to_user_id": 0,
      "canceled": false
    }
  ]
}
```

- `user_id`: Max ID is 10,000
- `currency`: One of "GBP", "USD", "RUB", "JPY", "CHF"
- `transactions`: Max count is 10
  - `amount`: Max value is 1,000
  - `to_user_id`: Target user ID
  - `canceled`: Can be omitted if false
- Fields order is not guaranteed

## Output

- Print the total amount of external (non-self) transactions converted to USD

## Constraints

- Optimize for speed (CPU time)
- Languages: C, C++, Go, Rust
