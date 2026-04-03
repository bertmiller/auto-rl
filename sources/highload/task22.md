# Task 22: Liquidation Engine (Trading Accounts)

**Source:** https://highload.fun/tasks/24

## Description

You are quoting derivatives prices which counterparties trade against, posting collateral in dollars. Counterparty accounts are initialized with a USD deposit, and they gain or lose value as prices move, in proportion to their position size in each instrument.

Accounts must maintain equity of at least 1% of their total position size across instruments. When prices update, you must find and liquidate any accounts in violation of this margin requirement.

## Key Calculations

- **Account equity** = balance + sum(size_i * price_i) - sum(total_paid_i)
- **Total position notional** = sum(|size_i| * price_i)
- **Margin rule**: if equity < total_notional / 100 after any price update, the account is liquidated

## Liquidation Priority

For any given price update, liquidate accounts with the largest total position notional first. As a tie-breaker, sort by account ID descending.

## Input Commands (via STDIN)

- `a <balance>` - Create account with given USD balance. IDs start at 0 and increment.
- `p <instrument_idx> <price>` - Add or update instrument price (0-indexed).
- `t <account_idx> <instrument_idx> <size>` - Trade size (signed) at the most recent price of that instrument.

## Output

- For each liquidated account, print: `liquidate <account_id> <equity> <position_notional>` to STDOUT
- Clear the liquidated account's balance and positions

## Constraints

- Optimize for speed (CPU time)
- Languages: C, C++, Go, Rust
