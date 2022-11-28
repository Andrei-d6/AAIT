from plot_spectrum import plot_spectrum
from plot_counts import plot_counts
from plot_eval_performance import read_crafter_logs


from pathlib import Path


BASE_OUPUT_PATH = Path('../plots')
LOGIDR = Path('../logdir')


def make_spectrum_plots():
    indirs = ['dqn_agent', 'random_agent', 'test_agents']
    indirs = list(map(lambda x: LOGIDR/x, indirs))

    labels = ['dqn_agent', 'random_agent', 'test_agent']
    colors = ['#377eb8', '#5fc35d', '#984ea3']

    outpath = BASE_OUPUT_PATH/'plot_achievements.png'

    plot_spectrum(indirs, labels, colors=colors, outpath=outpath)


def make_count_plots():

    plot_counts(LOGIDR/'dqn_agent',
                BASE_OUPUT_PATH/'count_dqn.png','#377eb8')

    plot_counts(LOGIDR/'random_agent',
                BASE_OUPUT_PATH/'count_random.png', '#984ea3')


def make_avg_reward_plots():

    indirs = ['dqn_agent', 'random_agent', 'test_agents']
    indirs = list(map(lambda x: LOGIDR/x, indirs))

    outpath = BASE_OUPUT_PATH/'plot_eval_rewards.png'

    read_crafter_logs(indirs, outpath=outpath, show=False)


def main():
    make_spectrum_plots()
    make_count_plots()
    make_avg_reward_plots()

if __name__ == '__main__':
    main()