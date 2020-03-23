import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
from scipy import stats
# tf.random.set_seed() Uncomment if you need a to use random seed

sims = int(input('Number of Simulations: \n'))


def calculate_revenue():
    # Transaction Number [Uniform]
    # Values have an equal probability of falling between 0 - 5000, the largest
    # being considered a fairly good amount of transactions.
    transaction_num_dist = tfd.Sample(tfd.Uniform(low=0, high=5000), sample_shape=1)
    transaction_num_sample = transaction_num_dist.sample().numpy().astype('int32')

    # Transaction Cost [Triangular]
    # Assumed that the values can go from $2.5 (snack or novelty item) to $18
    # for something a bit more expensive, with the most likely value being $7
    transaction_cost_dist = tfd.Sample(tfd.Triangular(low=2.5, high=18, peak=7), sample_shape=1)
    transaction_cost_sample = transaction_cost_dist.sample().numpy()

    # Revenue sample minus the per transaction costs and the marketing allocation
    revenue_pre = transaction_cost_sample * transaction_num_sample

    # Marketing budget for the month [Normal]
    # Average percent cut of revenue for marketing will be 18% (loc), with a
    # standard deviation of 2.5% (scale), meaning that about 68% of values will
    # range from 15.5% to a more aggresive 20.5%.
    mkt_dist = tfd.Sample(tfd.Normal(loc=0.18, scale=0.025), sample_shape=1)
    mkt_sample = mkt_dist.sample().numpy()

    # Credit Card/PayPal cut [Discrete]
    # Both payment servicers charge 2.4% or 2.9% per transaction, so they each
    # get a 50/50 chance assuming no cash is accepted
    # The discrete distribution is implemented differently because there wasn't
    # one in the TFP libraries. This is how you would use it with scipy.stats
    card_cut_vals = np.array([0.024, 0.029])
    card_cut_probs = (0.50, 0.50)
    # This is done because rv_discrete only takes integers as the values
    card_cut_dist = stats.rv_discrete(values=(range(len(card_cut_probs)), card_cut_probs))
    idx_card_cut = card_cut_dist.rvs(size=1)
    card_cut_sample = card_cut_vals[idx_card_cut]

    revenue = revenue_pre - (revenue_pre * mkt_sample) - (revenue_pre * card_cut_sample)
    return np.around(revenue, decimals=2)


def calculate_cost():
    # Fixed variables [Constants]
    incorp_MA_cost = 500 # Where I was located at the time
    domain_regist = 29.99 # GoDaddy prices at the time

    # Reasonable salaries given the perceived staff needs
    dev_salary_mo = round(100000, 2) * 2
    ops_manager_salary_mo = round(75000, 2)
    hr_salary_mo = round(60000, 2)
    exec_salary_mo = round(50000, 2) * 2

    # Constants added together and divided by 12 months
    fixed_vars_total = (incorp_MA_cost + domain_regist + dev_salary_mo + ops_manager_salary_mo + hr_salary_mo + exec_salary_mo)/12

    # Web hosting/month [Discrete]
    # $24.99 and $34.99 were likely values at the time based on traffic
    # The discrete distribution
    web_host_vals = np.array([24.99, 34.99])
    web_host_probs = (0.50, 0.50)
    web_host_dist = stats.rv_discrete(values=(range(len(web_host_probs)), web_host_probs))
    idx_web_host = web_host_dist.rvs(size=1)
    web_host_sample = web_host_vals[idx_web_host]

    # Liability Insurance/month [TruncNormal]
    # This is a bounded Normal distribution, with mean = 40 and SD = 20 but
    # between $25-120
    insurance_liab_dist = tfd.Sample(tfd.TruncatedNormal(loc=40, scale=20, low=25, high=120), sample_shape=1)
    insurance_liab_sample = insurance_liab_dist.sample().numpy()

    cost = fixed_vars_total + web_host_sample + insurance_liab_sample
    return np.around(cost, decimals=2)

def calculate_investment_capital():

    # Amount of capital invested for a year [Triangular]
    # Assume the amount invested goes from $5000-1M, with the most likely values
    # being $250K
    investment_dist = tfd.Sample(tfd.Triangular(low=50000, high=1000000, peak=250000), sample_shape=1)
    investment_sample = investment_dist.sample().numpy()

    return np.around(investment_sample, decimals=2)

# Set up to simulate periods of one year per iteration
def monte_carlo(simulations=sims):
    tax_rate = 0.30 # Just an approximation, can follow the U.S. brackets to be more realistic
    flow_months = [] # List to save each 12 months before calculating into one simulation
    cash_flow_output = []
    month_counter = 0
    capital = calculate_investment_capital()/12

    for i in range(simulations):
        for month in range(12):
            revenue = calculate_revenue()
            cost = calculate_cost()
            taxes = (revenue - cost) * tax_rate

            if month_counter == 11: # So that it resets after 12 months
                month_counter = 0
                capital = calculate_investment_capital()/12
                cash_flow_output.append(np.sum(flow_months))
                flow_months = []

            total = (revenue - cost - taxes + capital) * 0.25 # User get's 75% cut
            flow_months.append(total)
            month_counter += 1

    return np.around(cash_flow_output, decimals=2)

def monte_carlo_plot_stats(sims):
    import matplotlib.pyplot as plt

    sim_results = monte_carlo(sims) # Runs the actual simulation
    sim_stats = stats.describe(sim_results)

    plt.hist(sim_results, bins=100)

    plt.xlabel('First Year Cash Flow [$]')
    plt.ylabel('Frequency')
    plt.title('Monte Carlo Simulation')

    plt.savefig(f'monte_carlo_{sims}iterations.png', dpi=300)
    return str(sim_stats)


result = monte_carlo_plot_stats(sims)
print(result)
