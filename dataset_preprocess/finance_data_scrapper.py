import wikipedia
import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime

def get_pages_from_category(category, max_pages=50):
    """Get pages from a specific category"""
    try:
        url = f"https://en.wikipedia.org/wiki/{category}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        pages = []
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if '/wiki/' in href and ':' not in href and '#' not in href:
                pages.append(href.split('/wiki/')[-1])

        return list(set(pages))[:max_pages]
    except Exception as e:
        print(f"Error fetching category {category}: {e}")
        return []


def get_page_content(title):
    """Get content from a Wikipedia page"""
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return {
            'title': title,
            'content': page.content,
            'url': page.url
        }
    except Exception as e:
        print(f"Error fetching page {title}: {e}")
        return None


def get_finance_categories():
    """Get expanded finance-related Wikipedia categories"""
    categories = [
        # Core Finance
        'Category:Banking',
        'Category:Financial_markets',
        'Category:Investment',
        'Category:Financial_economics',
        'Category:Public_finance',
        'Category:Financial_risk',
        'Category:Corporate_finance',

        # Debt-specific Categories
        'Category:Debt',
        'Category:Credit',
        'Category:Government_debt',
        'Category:Bonds_(finance)',
        'Category:Interest_rates',
        'Category:Loans',
        'Category:Mortgage',
        'Category:Consumer_debt',
        'Category:Bankruptcy',
        'Category:Debt_collection',

        # Additional Financial Topics
        'Category:Financial_ratios',
        'Category:Stock_market',
        'Category:Derivatives_(finance)',
        'Category:Risk_management',
        'Category:Financial_regulation',
        'Category:Accounting',
        'Category:Financial_planning',
        'Category:Portfolio_theories',
        'Category:Asset_management',
        'Category:Venture_capital',
        'Category:Private_equity',
        'Category:Hedge_funds',
        'Category:Financial_technology'
    ]
    return categories


def get_investopedia_content(term):
    """Get content from Investopedia (simulated)"""
    try:
        # Simulated Investopedia content structure
        definitions = {
            'Debt': """Debt is money borrowed by one party from another. Forms of debt include:
                    1. Secured Debt: Backed by collateral
                    2. Unsecured Debt: Not backed by collateral
                    3. Revolving Debt: Credit cards and lines of credit
                    4. Installment Debt: Regular payment schedules

                    Key Calculations:
                    - Debt-to-Income Ratio = Total Monthly Debt Payments / Gross Monthly Income
                    - Debt Service Coverage Ratio = Net Operating Income / Total Debt Service""",
            'Interest': """Interest is the cost of borrowing money. Calculations include:

                    Simple Interest = Principal × Rate × Time
                    Compound Interest = Principal × (1 + Rate)^Time - Principal

                    Example:
                    $1000 borrowed at 5% for 3 years
                    Simple Interest = $1000 × 0.05 × 3 = $150
                    Compound Interest = $1000 × (1 + 0.05)^3 - $1000 = $157.63""",
            'Bond_Valuation': """Bond valuation determines the theoretical fair value of a bond. Key concepts:

                    Bond Price = Sum of PV of all coupons + PV of par value

                    Duration = Σ(t × PVt) / Price
                    where t = time to cash flow, PVt = present value of cash flow

                    Example:
                    5-year bond, 5% coupon, $1000 par value
                    Semi-annual payments: $25 each
                    Market rate: 6%"""
        }
        return definitions.get(term, "")
    except Exception as e:
        print(f"Error fetching Investopedia content for {term}: {e}")
        return ""


def add_calculations_and_examples(content):
    """Add relevant financial calculations and examples"""
    calculations = {
        'Debt': """
        Common Debt Calculations:

        1. Loan Payment (PMT):
        PMT = PV * (r(1+r)^n)/((1+r)^n-1)
        where:
        PV = Present Value
        r = Interest Rate per period
        n = Number of periods

        2. Total Interest Paid:
        Total Interest = (PMT × n) - PV

        3. Amortization Schedule Example:
        $200,000 mortgage, 30 years, 4% annual interest
        Monthly payment = $954.83
        First payment breakdown:
        - Principal: $288.16
        - Interest: $666.67
        """,

        'Risk': """
        Risk Metrics:

        1. Value at Risk (VaR):
        95% VaR = μ - (1.645 × σ)
        where:
        μ = expected return
        σ = standard deviation

        2. Sharpe Ratio:
        Sharpe = (Rp - Rf) / σp
        where:
        Rp = Portfolio return
        Rf = Risk-free rate
        σp = Portfolio standard deviation
        """
    }

    # Add calculations if keywords found in content
    for key, calc in calculations.items():
        if key.lower() in content.lower():
            content += f"\n\nCalculations and Examples:\n{calc}"

    return content


def create_finance_dataset(output_file='enhanced_finance_dataset.txt', max_size_mb=15):
    """Create an enhanced dataset of finance-related content"""
    categories = get_finance_categories()
    all_text = ""

    # Add dataset metadata
    all_text += f"""=== Financial Dataset ===
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Contents: Financial theory, calculations, definitions, and practical examples
Topics: Banking, Debt, Investments, Risk Management, and more\n\n"""

    for category in categories:
        print(f"Processing category: {category}")
        pages = get_pages_from_category(category, max_pages=100)  # Increased max pages

        for page in pages:
            # Get Wikipedia content
            wiki_content = get_page_content(page)
            if wiki_content:
                content = wiki_content['content']

                # Add Investopedia content if available
                investopedia_content = get_investopedia_content(page)
                if investopedia_content:
                    content += f"\n\nAdditional Information:\n{investopedia_content}"

                # Add calculations and examples
                content = add_calculations_and_examples(content)

                # Add to dataset with clear section formatting
                all_text += f"\n{'=' * 80}\n"
                all_text += f"TOPIC: {wiki_content['title']}\n"
                all_text += f"SOURCE: {wiki_content['url']}\n"
                all_text += f"{'=' * 80}\n\n"
                all_text += content + "\n\n"

                # Check file size
                size_mb = len(all_text.encode('utf-8')) / (1024 * 1024)
                print(f"Current dataset size: {size_mb:.2f} MB")
                if size_mb >= max_size_mb:
                    break

            time.sleep(random.uniform(1, 2))  # Randomized delay

        if len(all_text.encode('utf-8')) / (1024 * 1024) >= max_size_mb:
            break

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(all_text)

    print(f"Dataset created: {output_file}")
    print(f"Final size: {len(all_text.encode('utf-8')) / (1024 * 1024):.2f} MB")

    return output_file


if __name__ == "__main__":
    dataset_file = create_finance_dataset()
