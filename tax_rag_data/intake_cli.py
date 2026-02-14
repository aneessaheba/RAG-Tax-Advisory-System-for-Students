import os
import json
from dataclasses import dataclass, asdict, field
from typing import List

def input_list(prompt):
    val = input(prompt)
    return [v.strip() for v in val.split(',') if v.strip()]

@dataclass
class UserTaxProfile:
    visa_type: str
    tax_year: str
    first_entry_year: str
    home_country: str
    income_types: List[str] = field(default_factory=list)
    state: str = ''

    def to_dict(self):
        return asdict(self)

def collect_profile():
    print("--- Tax Profile Intake ---")
    visa_type = input("Visa type (e.g., F-1, J-1): ").strip()
    tax_year = input("Tax year (e.g., 2025): ").strip()
    first_entry_year = input("First entry year (e.g., 2023): ").strip()
    home_country = input("Home country: ").strip()
    income_types = input_list("Income types (comma-separated, e.g., scholarship, wage): ")
    state = input("State (e.g., CA): ").strip()
    return UserTaxProfile(
        visa_type=visa_type,
        tax_year=tax_year,
        first_entry_year=first_entry_year,
        home_country=home_country,
        income_types=income_types,
        state=state
    )

def save_profile(profile, path='user_profile.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"Profile saved to {path}")

def main():
    profile = collect_profile()
    save_profile(profile)
    print("Profile intake complete. Attach this profile to each query context.")

if __name__ == "__main__":
    main()
