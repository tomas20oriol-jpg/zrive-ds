# Module 3: TRD — Project Requirements Document
## Push Notifications

---

## Context & Goal

Our customers can make purchases in our groceries store via the company app. From time to time there are items we want to promote: it could be that we are selling them out to discontinue them, it could be there are lots about to expire, or we may be seeking to increase market share over them.

Sending push notifications to our users is an effective manner to boost sales and offer discounts to incentivize user engagement with our targeted products. However, push notifications may be intrusive and sending too many can generate user dissatisfaction and churn — i.e. users uninstall our app — which is a significant cost for us.

> Current push notifications have an open rate of around **5%** in our app.

We want to build a product that relies on a **predictive model** that allows us to target a set of users highly likely to be interested in a chosen item, and send them a push notification to promote it.

---

## Requirements

- We are only interested in users that purchase the promoted item as part of a basket of **at least 5 items**, since shipping costs for a single or few items can exceed the gross margin.
- The system should allow sales operators to:
  - Select an item from a **dropdown or search bar**
  - Get the **segment of users** to target
  - **Trigger a customizable push notification**

---

## Planning

This is a **high priority** tool and our competitors are moving fast in a similar direction.

| Milestone | Timeline |
|---|---|
| Proof of Concept (PoC) | 1 week |
| Go Live | 2–3 weeks |

---

## Impact

The target impact is:

- **+2%** increase in monthly sales
- **+25%** boost over selected promoted items

> For more details on these figures, please refer to the **Sales Department Push Analysis Report**.