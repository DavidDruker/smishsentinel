"""Builds a batch of hand-authored 'hard ham' TRAINING examples -- legitimate
transactional/security messages (OTP codes, bank alerts, delivery, etc.)
that share surface vocabulary with scams. Committed for the same
transparency reason as train_screener.py/build_training_data.py.

Deliberately a SEPARATE, non-overlapping set from the 316-message benchmark
in research_dump/benchmark/hard_ham_v3.json: that set is reserved for
evaluation. Training on it directly would contaminate the exact benchmark
used to measure whether this retraining worked. This file covers the same
categories with different brands, amounts, and phrasing -- verified against
the benchmark for exact and near-duplicate overlap before being folded into
the training corpus (see append_to_training_data.py).
"""
import json

MESSAGES = [
    # --- 2FA / OTP verification codes ---
    ("otp", "Your Apple ID code is 384719. Do not share this with anyone, including Apple support."),
    ("otp", "Netflix: your temporary access code is 205917."),
    ("otp", "Your Chase verification code is 719044. Valid for 10 minutes."),
    ("otp", "eBay: your security code is 601248. Never share this code with anyone."),
    ("otp", "Your Reddit login code is 88213."),
    ("otp", "BMO: your one-time passcode is 552901 for online banking access."),
    ("otp", "Your Twitch code is 71904."),
    ("otp", "Ubisoft: your verification code is 205611."),
    ("otp", "Your Samsung account code is 449021. Expires in 10 minutes."),
    ("otp", "Scotiabank: your temporary access PIN is 673120."),
    ("otp", "Your Slack workspace code is 90142."),
    ("otp", "DoorDash: your verification code is 55102."),
    ("otp", "Your Dropbox code is 208715. Don't share this with anyone."),
    ("otp", "Wealthsimple Trade: your login code is 671829."),
    ("otp", "Your Spotify code is 330central12."),

    # --- Delivery / shipping notifications ---
    ("delivery", "Purolator: Your parcel is 3 stops away on today's route."),
    ("delivery", "Canada Post: Your package was successfully delivered at 2:41pm to your front door."),
    ("delivery", "Amazon: Your order of kitchen supplies has shipped, arriving Thursday."),
    ("delivery", "UPS: Your delivery window today is 1pm-5pm."),
    ("delivery", "Loblaws: Your grocery delivery is 20 minutes away."),
    ("delivery", "FedEx: Your package cleared customs this morning and is now in transit."),
    ("delivery", "Canadian Tire: Your curbside order is ready for pickup at door 4."),
    ("delivery", "Well.ca: Your order has shipped and should arrive within 2 business days."),
    ("delivery", "Amazon: Your delivery was attempted but a gate code is needed. Update it in your account."),
    ("delivery", "Staples: Your print order is ready for pickup at the Yonge St. location."),
    ("delivery", "Pizza Pizza: Your order is out for delivery, ETA 15 minutes."),
    ("delivery", "London Drugs: Your online order is ready for in-store pickup."),
    ("delivery", "Uber Eats: The restaurant is preparing your order now."),
    ("delivery", "Canada Post: Your parcel was too large for your mailbox, it's at the post office for pickup."),

    # --- Appointment / service reminders ---
    ("appointment", "Reminder: Your teeth cleaning with Dr. Osei is booked for 11am tomorrow."),
    ("appointment", "Your registration renewal appointment at ServiceOntario is confirmed for 10am Thursday."),
    ("appointment", "Reminder: Your kids' pediatrician checkup is Friday at 2:30pm."),
    ("appointment", "Your window cleaning service is scheduled for tomorrow morning."),
    ("appointment", "Reminder: Your therapy session with Dana is today at 4pm via video call."),
    ("appointment", "Your carpet cleaning appointment is confirmed for Saturday 9am-11am."),
    ("appointment", "Reminder: Your annual eye exam is next week, Tuesday at 1pm."),
    ("appointment", "Your dishwasher repair technician arrives between 12-4pm tomorrow."),
    ("appointment", "Reminder: Your driving test is booked for Monday at 9:15am, please arrive early."),
    ("appointment", "Your pet grooming appointment for Bella is confirmed for 3pm Wednesday."),
    ("appointment", "Reminder: Your quarterly HVAC maintenance visit is scheduled for Friday."),
    ("appointment", "Your consultation with the financial advisor is confirmed for 2pm Thursday."),

    # --- Legitimate bank / financial alerts ---
    ("bank_alert", "Scotiabank Alert: A purchase of $23.50 was made at Tim Hortons. Call the number on your card if unrecognized."),
    ("bank_alert", "BMO: Your paycheque of $2,104.55 has been deposited."),
    ("bank_alert", "CIBC Fraud Alert: We flagged a transaction for review. Call the number on your card to confirm or deny it."),
    ("bank_alert", "Tangerine: Your bill payment of $210.00 to Bell was completed."),
    ("bank_alert", "RBC: Your e-transfer of $60.00 to Jamie R. has been sent."),
    ("bank_alert", "Simplii Financial: Your monthly statement is now ready to view online."),
    ("bank_alert", "TD: A withdrawal of $300 was made at a branch ATM. Contact us using the number on your card if this wasn't you."),
    ("bank_alert", "Desjardins: Your automatic savings transfer of $100 was completed as scheduled."),
    ("bank_alert", "Manulife Bank: Your mortgage renewal documents are ready for review online."),
    ("bank_alert", "Interac e-Transfer: You've received $150.00 from Taylor B."),
    ("bank_alert", "CIBC: Your credit card payment due date has been updated, see your statement for details."),
    ("bank_alert", "National Bank: Your investment account statement for August is now available."),
    ("bank_alert", "RBC: We noticed a login from a new location. If this was you, no action is needed."),

    # --- Legitimate opt-in marketing / promotions ---
    ("marketing", "Gap: Buy one get one 50% off this weekend only. Reply STOP to unsubscribe."),
    ("marketing", "Chapters Indigo: Members get early access to the fall sale starting tonight. STOP to opt out."),
    ("marketing", "Foot Locker: New sneaker drop this Friday. Text STOP to unsubscribe."),
    ("marketing", "Marshalls: Clearance racks just got bigger. Reply STOP anytime to unsubscribe."),
    ("marketing", "PetSmart: Dog food sale this week only. Text STOP to opt out."),
    ("marketing", "Staples: Back to school deals end Sunday. Reply STOP to unsubscribe."),
    ("marketing", "H&M: New arrivals are here. Shop the collection online. Text STOP to opt out."),
    ("marketing", "Canadian Tire: Your points are worth extra this weekend. STOP to unsubscribe."),
    ("marketing", "Sobeys: Load your digital coupons before they expire Sunday. Reply STOP to opt out."),
    ("marketing", "Golf Town: End of season clearance starts today. Text STOP to unsubscribe."),
    ("marketing", "Toys R Us: Holiday catalogue is live online now. Reply STOP to opt out."),
    ("marketing", "Bath & Body Works: Semi-annual sale starts tomorrow. Text STOP to unsubscribe."),

    # --- Utility / bill / account notifications ---
    ("utility", "Hydro One: Your bill of $103.44 is ready to view online."),
    ("utility", "Bell Mobility: Your data add-on was applied successfully."),
    ("utility", "Amazon Music: Your subscription renews on the 5th for $10.99."),
    ("utility", "Cogeco: Your internet bill of $79.99 is due on the 20th."),
    ("utility", "YouTube: Your channel membership payment was processed successfully."),
    ("utility", "Rogers Ignite: Your equipment return has been confirmed, thank you."),
    ("utility", "Rara (formerly Freedom): Your plan renews next week at the same rate."),
    ("utility", "Apple Music: Your family plan renews tomorrow for $16.99."),
    ("utility", "EPCOR: Your water bill of $58.20 is now available online."),
    ("utility", "Bell Fibe: Your channel package was updated as requested."),

    # --- Workplace / school / community notifications ---
    ("community", "Reminder: Staff meeting moved to 2pm today in the main boardroom."),
    ("community", "Ottawa-Carleton District School Board: Early dismissal today at 1pm due to weather."),
    ("community", "Reminder: Curbside composting starts next week, bins will be delivered Monday."),
    ("community", "Your gym's pool is closed for maintenance this weekend, reopening Monday."),
    ("community", "Reminder: The strata AGM is next Tuesday at 7pm in the amenity room."),
    ("community", "Halton Region: Watering restrictions are lifted as of today."),
    ("community", "Reminder: Your child's swim class was moved to the shallow pool this week."),
    ("community", "Mississauga Transit: Route 5 is on detour this week due to construction."),
    ("community", "Reminder: Volunteer orientation is this Saturday at 10am at the shelter."),

    # --- Legitimate loyalty / rewards notices ---
    ("loyalty", "Metro & Moi: You've earned bonus points on your grocery haul this week."),
    ("loyalty", "Choice Hotels: You're one stay away from your next free night."),
    ("loyalty", "Avios: Your points balance has been updated after your recent flight."),
    ("loyalty", "Circle K: You've earned a free coffee with your fuel purchase today."),
    ("loyalty", "IKEA Family: You have a reward voucher ready to use on your next visit."),
    ("loyalty", "Best Buy Rewards: You've earned $5 in reward certificates."),

    # --- Legitimate customer-service replies ---
    ("support", "Cogeco: Your service ticket has been closed, thank you for your patience."),
    ("support", "eBay: Your return has been approved, print your label from the app."),
    ("support", "Samsung: Your device repair is complete and ready for pickup."),
    ("support", "Shaw: A technician has been dispatched, ETA is 45 minutes."),
    ("support", "Sobeys: We're sorry about the substitution, a refund has been issued."),
    ("support", "Staples: Your print job had an issue, we've reprinted it at no charge."),

    # --- Travel / event confirmations ---
    ("travel", "WestJet: Check-in opens in 22 hours for your flight to Calgary."),
    ("travel", "Airbnb: Your check-in instructions have been sent by your host."),
    ("travel", "Flair Airlines: Your flight has been confirmed, seat 12A."),
    ("travel", "Live Nation: Your presale code is now active for tomorrow's on-sale."),
    ("travel", "Enterprise: Your rental car reservation is confirmed for pickup Friday at noon."),
    ("travel", "GO Transit: Your monthly pass renews automatically on the 1st."),

    # --- Insurance / warranty notices ---
    ("insurance", "Aviva: Your home insurance renewal is ready to review online."),
    ("insurance", "Canada Life: Your benefits statement for this quarter is now available."),
    ("insurance", "Economical Insurance: Your claim adjuster will contact you within 2 business days."),
    ("insurance", "RBC Insurance: Your travel insurance policy has been emailed to you."),
    ("insurance", "Definity: Your auto quote is ready to review in your account."),

    # --- Survey / feedback requests ---
    ("survey", "Sobeys: Tell us how we did today, survey link sent to your email."),
    ("survey", "WestJet: How was your flight? Rate your experience in the app."),
    ("survey", "Staples: We'd love your feedback on today's visit, no link needed, just reply."),
    ("survey", "London Drugs: Quick 1-minute survey about your pharmacy visit, available at checkout."),

    # --- School / parent notifications ---
    ("school", "Reminder: Report card pickup is this Thursday from 3-6pm."),
    ("school", "Waterloo Region District School Board: Winter break starts December 19th."),
    ("school", "Reminder: Your child's field trip payment is due by Friday."),
    ("school", "Reminder: Spirit week starts Monday, see the schedule on the school app."),
    ("school", "McMaster University: Your fall semester grades are now posted."),

    # --- Political / civic notices ---
    ("political", "Reminder: Municipal budget consultation meeting is Wednesday at 7pm, all welcome."),
    ("political", "This is your local school trustee's office with a reminder about tonight's forum."),
    ("political", "Elections Ontario: Voter information cards were mailed this week."),

    # --- Prescription / pharmacy reminders ---
    ("prescription", "London Drugs Pharmacy: Your prescription is ready, pickup available until 9pm."),
    ("prescription", "Reminder: Your flu shot appointment at the clinic is tomorrow at 11am."),
    ("prescription", "Pharmacy: Your medication renewal was approved by your doctor."),
    ("prescription", "Costco Pharmacy: Time to refill, reply REFILL or call the pharmacy directly."),

    # --- Rental / property notices ---
    ("rental", "Reminder: Your storage unit payment is due on the 1st."),
    ("rental", "Your building's elevator will be out of service for maintenance Tuesday."),
    ("rental", "Property management: Your parking pass renewal is ready for pickup at the office."),
    ("rental", "Reminder: Move-in inspection is scheduled for Saturday at 10am."),

    # --- Charity / donation notices ---
    ("charity", "Thank you for your donation to the animal shelter, your receipt has been emailed."),
    ("charity", "Covenant House: Thank you for supporting youth this month."),
    ("charity", "Reminder: Our charity run registration closes Friday, thanks for signing up."),

    # --- Job application / hiring updates ---
    ("job_application", "Thank you for applying to the Junior Analyst role, we'll be in touch within 10 days."),
    ("job_application", "Your phone screen is confirmed for Wednesday at 1pm."),
    ("job_application", "Glassdoor: A company you follow just posted a new opening."),
    ("job_application", "Thank you for the great conversation today, HR will follow up by end of week."),

    # --- Personal / casual conversational ---
    ("personal", "Running late, the bus never showed up"),
    ("personal", "Can you pick up the kids today? Something came up at work"),
    ("personal", "That movie was so good, we should watch the sequel next"),
    ("personal", "Left the oven on I think, can you check when you get home?"),
    ("personal", "Happy anniversary! Can't believe it's been 5 years"),
    ("personal", "Are we still doing game night Friday?"),
    ("personal", "Just got back from the trip, exhausted but it was worth it"),
    ("personal", "Can you send me the wifi password again? Forgot it"),
    ("personal", "So sorry for your loss, thinking of you"),
    ("personal", "Practice starts at 6 tonight, don't forget your cleats"),
]

cases = [
    {"id": f"trainham_{i:04d}", "label": "ham", "category": category, "text": text}
    for i, (category, text) in enumerate(MESSAGES)
]

from collections import Counter
print(f"Total new hard-ham training messages: {len(cases)}")
print(f"By category: {dict(Counter(c['category'] for c in cases))}")

with open("hard_ham_training_v1.json", "w", encoding="utf-8") as f:
    json.dump(cases, f, indent=2)
print("Saved to hard_ham_training_v1.json")
