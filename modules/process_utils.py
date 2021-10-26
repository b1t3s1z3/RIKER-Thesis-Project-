from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd


# Reads a log file of time stamps that indicate when the PokerTH log file was modified (i.e. when an action was taken
# in the game).
def get_timestamps(filename: str, additional_stamps: int) -> list[str]:
    timestamps = []

    with open(filename) as f:
        for line in f:
            if "Modified file" in line:
                for word in line.split():
                    if "," in word:
                        # print(word)
                        timestamps.append(word)

        # Extra lines added for overflow in timestamps matching with PokerActs later...
        for i in range(additional_stamps):
            timestamps.append("00:00:00,000")

    return timestamps


labels = ['frame', 'timestamp', 'confidence', 'success', 'AU01', 'AU02',  # names of the columns
          'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14',
          'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU45']


# Reads in the data from a .csv file as output by OpenFace FeatureExtraction with the '-aus' flag on.
def parse_csv(filename: str) -> np.ndarray:
    load_cols = [0] + [i for i in range(2, 22)]  # which columns we want to use
    load_types = [np.int32, np.float16, np.float32, np.int32] + [np.float32] * 17  # types of the columns

    load_file = np.genfromtxt(
        fname=filename,
        usecols=load_cols,
        dtype=load_types,
        skip_header=1,
        delimiter=',',
        names=labels
    )

    return load_file


def plot_raw_data(raw_data: pd.DataFrame) -> None:
    plt.figure(figsize=(25, 10))
    for label in labels:
        plt.plot(raw_data['frame'], raw_data[label])
    plt.legend()


def data_smoothed(t: np.ndarray, win=5) -> np.ndarray:
    smoothed_au = np.vstack(tuple(map(lambda label: np.correlate(t[label], np.ones(win) / win, 'same'), labels)))
    return smoothed_au


def au_detect(action_unit: np.ndarray, label: str) -> pd.DataFrame:
    # ***COMMENT NEEDS UPDATE***An array containing codes for noting when gestures started(1), peaked(2),
    # and ended(3) during the sequence, as well as the amplitude for the that gesture (encoded as the amplitude
    # change from start to peak), and the duration of the gesture (encoded as the distance between the start and the
    # end indices).

    au_data = np.array([0, 0, 0, 0, 0]).reshape((1, 5))
    au_df = pd.DataFrame(au_data, columns=list('SPEBT'))
    au_df['AU Label'] = [label]
    # print(au_df)

    begun = False
    rising = False
    peaked = False
    ending = False
    sensitivity = 0.07
    start_index = 0
    start_amp = 0
    peak_index = 0
    peak_amp = 0

    for i in range(len(action_unit)):
        if i > 1:
            if action_unit[i] - action_unit[i - 2] >= sensitivity and rising is False:
                rising = True
                begun = True
                output = "Started a gesture at " + repr(i)
                start_amp = action_unit[i - 2]
                next_au = np.array([start_index, peak_index, i - 1, start_amp, peak_amp]).reshape((1, 5))
                next_au_df = pd.DataFrame(next_au, columns=list('SPEBT'))
                next_au_df['AU Label'] = [label]
                new_list = au_df, next_au_df
                au_df = pd.concat(new_list)
                start_index = i
                peak_index = 0
            elif action_unit[i] - action_unit[i - 2] >= sensitivity and rising is True:
                output = "Kept going up at " + repr(i)
            elif action_unit[i] + action_unit[i - 1] < sensitivity and rising is False and ending is False:
                output = "Still at baseline at " + repr(i)
            elif action_unit[i] - action_unit[i - 1] <= sensitivity and rising is True:
                rising = False
                peaked = True
                ending = False
                output = "Peaked at " + repr(i)
                peak_index = i
                peak_amp = action_unit[i - 1]
            elif action_unit[i] - action_unit[i - 2] <= (0 - sensitivity) and rising is False and peaked is True:
                rising = False
                peaked = True
                ending = True
                output = "Coming down at " + repr(i - 1)
            elif action_unit[i] - action_unit[i - 2] < sensitivity < action_unit[i] and ending is True:
                output = "Held steady for now at " + repr(i - 1)
            elif action_unit[i] - action_unit[i - 2] < sensitivity and ending is True and action_unit[i] < sensitivity:
                begun = False
                rising = False
                peaked = False
                ending = False
                output = "Back to baseline at " + repr(i)
                next_au = np.array([start_index, peak_index, i - 1, start_amp, peak_amp]).reshape((1, 5))
                next_au_df = pd.DataFrame(next_au, columns=list('SPEBT'))
                next_au_df['AU Label'] = [label]
                new_list = au_df, next_au_df
                au_df = pd.concat(new_list)
                start_index = i
                peak_index = i
                peak_amp = 0

    return au_df


# Combine the facial actions into a single dataframe.
def aggregate_actions(smoothed_actions: np.ndarray) -> pd.Series:
    au_labels = ['01', '02', '04', '05', '06', '07', '09', '10', '12', '14', '15', '17', '20', '23', '25', '26', '45']

    action_data = np.array([0, 0, 0, 0, 0]).reshape((1, 5))
    action = pd.DataFrame(action_data,
                          columns=['S', 'P', 'E', 'B', 'T'])  # Start Time, Peak Time, End Time, Bottom Amp, Top Amp
    action['AU Label'] = [0]
    action['Index'] = 0

    for i in range(len(smoothed_actions)):
        all_actions = au_detect(smoothed_actions[i], au_labels[i])
        new_list = action, all_actions
        action = pd.concat(new_list)

    # Remove actions which End at frame 0.
    action = action[action.E != 0]

    # Add the index column
    index_list = list(range(0, len(action['Index'])))
    for i in range(len(action['Index'])):
        action.iloc[i, 3:4] = index_list[i]
    action.set_index('Index', inplace=True)

    # Add the TotalAmp column
    action['TotalAmp'] = 0

    for i in range(len(action)):
        float_index = float(i)
        action.iloc[i, 6:7] = action.at[float_index, 'T'] - action.at[float_index, 'B']

    # Add a column containing the timing of the Peak in seconds
    action['Inflection'] = 0
    for i in range(len(action)):
        float_index = float(i)
        inflection_point = (action.at[float_index, 'P'])
        action.iloc[i, 7:8] = inflection_point / 30

    # Add a column containing the duration of action onset
    action['Onset Frame Count'] = 0
    for i in range(len(action)):
        float_index = float(i)
        onset_length = (action.at[float_index, 'P']) - (action.at[float_index, 'S'])
        action.iloc[i, 8:9] = onset_length

    # Add a column containing the duration of the action offset
    action['Offset Frame Count'] = 0
    for i in range(len(action)):
        float_index = float(i)
        offset_length = (action.at[float_index, 'E']) - (action.at[float_index, 'P'])
        action.iloc[i, 9:10] = offset_length

    # Drop the actions with a TotalAmp < 0
    action['Drop'] = 0
    for i in range(len(action)):
        float_index = float(i)
        if action.at[float_index, 'TotalAmp'] < 0.01:
            action.at[float_index, 'Drop'] = 1
    action = action[action.Drop != 1]

    # Sort by peaks of data
    peaks_sort = action.sort_values(['P', 'AU Label'], ascending=[True, True])

    return peaks_sort


# Adds the Gesture label to a dataframe of facial actions, so we can tell which gesture each action belongs to.
def add_gestures(actions: pd.Series, gesture_group_labels: np.ndarray) -> pd.Series:
    actions['Gesture'] = 0

    for i in range(len(gesture_group_labels)):
        actions.iloc[i, 11:12] = gesture_group_labels[i]

    return actions


# Creates a dataframe of gestures which is built from a dataframe of actions.
def actions_to_gestures(actions: pd.Series, action_clusters: int) -> pd.DataFrame:
    # Setting up new dataframe to receive FGs and metadata
    all_gestures = pd.DataFrame(index=range(0, action_clusters),
                                columns=['01', '02', '04', '05', '06', '07', '09', '10', '12', '14', '15', '17',
                                         '20', '23', '25', '26', '45'], dtype='float')
    all_gestures['Inflection'] = 0
    all_gestures['Onset Length'] = 0  # Mean number of frames of action onsets
    all_gestures['Onset Unity'] = 0  # Variance in frames of action onsets
    all_gestures['Offset Length'] = 0  # Mean number of rames of action offsets
    all_gestures['Offset Unity'] = 0  # Variance in frames of action offsets

    # For each Gesture
    for i in range(action_clusters):
        gesture_df = actions[actions.Gesture == i].set_index('AU Label')

        # Add the mean inflection point(peak point) for the gesture to the df
        inflection_point = gesture_df['Inflection'].mean()
        # print("Inflection point is " + repr(inflection_point))
        all_gestures.iloc[i, 17:18] = inflection_point

        # Add the onset length
        onset_length = gesture_df['Onset Frame Count'].mean()
        all_gestures.iloc[i, 18:19] = onset_length

        # Add the onset unity
        onset_unity = gesture_df['Onset Frame Count'].var()
        all_gestures.iloc[i, 19:20] = onset_unity

        # Add the offset length
        offset_length = gesture_df['Offset Frame Count'].mean()
        all_gestures.iloc[i, 20:21] = offset_length

        # Add the onset unity
        offset_unity = gesture_df['Offset Frame Count'].var()
        all_gestures.iloc[i, 21:22] = offset_unity

        # For each action unit observed
        for j in range(len(gesture_df)):
            # For each column in all_gestures, except 'Inflection'
            for k in range(len(all_gestures.columns) - 1):
                pos = k
                col = all_gestures.columns[pos]
                if col in gesture_df.index:
                    all_gestures.at[i, col] = gesture_df.iloc[0, 5:6]
            if len(gesture_df) > 1:
                gesture_df = gesture_df.iloc[1:]

    # Fill in missing values with 0s
    all_gestures_filled = all_gestures.fillna(0)

    # Total Amp of all Actions in the Gesture
    all_gestures_filled['SumAmp'] = all_gestures_filled.sum(axis=1)

    # Fix the SumpAmp so it doesn't include the Inflection Point (and other metadata) in the sum
    for i in range(len(all_gestures_filled)):
        all_gestures_filled.iloc[i, 22:23] = all_gestures_filled.iat[i, 22] - all_gestures_filled.iat[i, 21] - \
                                             all_gestures_filled.iat[i, 20] - all_gestures_filled.iat[i, 19] - \
                                             all_gestures_filled.iat[i, 18] - all_gestures_filled.iat[i, 17]

    return all_gestures_filled


# Removes gestures from a gestures dataframe which have very small amplitudes.
def gesture_filter_low(gestures: pd.DataFrame) -> pd.DataFrame:
    gestures['Drop'] = 0

    # Identify which gestures have SumAmps < 0.2...
    for i in range(len(gestures)):
        if gestures.at[i, 'SumAmp'] < 0.2:
            gestures.iloc[i, 23:24] = 1

    # ...and drop them.
    gestures = gestures[gestures.Drop != 1]

    # Reset the Index so it is still continuous
    gestures.reset_index(inplace=True)
    gestures = gestures.loc[:, '01':'SumAmp']

    return gestures


# Identifies the AUs that start and peak at similar times, and clusters them together.
def dbscan_propagate(au_smoothed: np.ndarray) -> (np.ndarray, int):
    # Gather all the Facial Actions into a single dataframe, sort that data frame by when the actions
    # peak in amplitude.
    sort_by_peaks = aggregate_actions(au_smoothed)
    peaks_sort = sort_by_peaks.sort_values(['P', 'AU Label'], ascending=[True, True])

    # Only count actions with a peak amplitude of greater than 0.
    x = peaks_sort.loc[peaks_sort['P'] >= 0.0, ['P', 'S']].values

    # Compute DBSCAN
    clustering = DBSCAN(eps=6.7, min_samples=2, n_jobs=-1).fit(x)
    cluster_centers_indices = clustering.core_sample_indices_
    labels_ = clustering.labels_
    n_clusters_ = len(cluster_centers_indices)

    # Plot clusters: UNCOMMENT THE FOLLOWING LINES UNTIL THE 'RETURN' LINE
    # IN ORDER TO GET A VISUAL GRAPHIC OF THE CLUSTERS. NOTE: THIS GRAPHIC
    # WILL LIKELY NOT BE VERY USEFUL FOR MORE THAN A FEW HUNDRED FRAMES OF
    # SUBJECT DATA – ANYTHING LONGER THAN THAT WILL JUST APPEAR TOO SMALL.
    # THIS IS BEST USED TO VALIDATE THAT THE FUNCTION IS WORKING PROPERLY
    # ON A SMALL/SHORT EXAMPLE DATASET.

    #     core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    #     core_samples_mask[clustering.core_sample_indices_] = True

    #     pp.close('all')
    #     pp.figure(figsize=(13,12))
    #    # Black removed and is used for noise instead.
    #     unique_labels = set(labels)
    #     colors = [pp.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    #     for k, col in zip(unique_labels, colors):
    #         if k == -1:
    #             # Black used for noise.
    #             col = [0, 0, 0, 1]

    #         class_member_mask = (labels == k)

    #         xy = X[class_member_mask & core_samples_mask]
    #         pp.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #                  markeredgecolor='k', markersize=3)

    #         xy = X[class_member_mask & ~core_samples_mask]
    #         pp.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #                  markeredgecolor='k', markersize=1)

    #     pp.title('Estimated number of clusters: %d' % n_clusters_)
    #     pp.show()

    return labels_, n_clusters_


def get_fg(subject_num: int) -> pd.DataFrame:
    raw_data = parse_csv(f'data/raw/face_only_{subject_num}.csv')
    # #plot_raw_data(raw_data)
    print("Data is imported")
    au_smoothed = data_smoothed(raw_data)
    print("Data is smoothed")
    actions = aggregate_actions(au_smoothed)
    print("Actions are aggregated into dataframe")
    # #print(actions)
    gesture_group_labels, n_clusters = dbscan_propagate(au_smoothed)
    # gesture_group_labels
    print("Gestures/clusters are identified")
    #     print("GESTURE GROUP LABELS ARE okay")
    #     # #print(gesture_group_labels)
    #     print("n_clusters is okay")
    # #print(n_clusters)
    act_with_gest = add_gestures(actions, gesture_group_labels)
    print("And actions have gesture labels.")
    # #print(act_with_gest)
    gestures_df = actions_to_gestures(act_with_gest, n_clusters)
    print("Gestures are in a data frame.")
    # #print(gestures_df)
    filtered_gestures = gesture_filter_low(gestures_df)
    print("Gestures are low-pass filtered.")

    return filtered_gestures


def trim_gestures(gestures: pd.DataFrame, timestamps: list[str], rec_start_time: datetime) -> pd.DataFrame:
    print("Trimming to fit game start and end time stamps.")
    game_start_time = datetime.strptime(timestamps[0], "%H:%M:%S,%f")
    game_end_time = datetime.strptime(timestamps[-6], "%H:%M:%S,%f")

    start_diff = game_start_time - rec_start_time
    end_diff = game_end_time - rec_start_time
    secs = start_diff.seconds
    mils = start_diff.microseconds
    end_secs = end_diff.seconds
    end_mils = end_diff.microseconds

    relative_start = float(secs + (mils / 1000000))
    relative_end = float(end_secs + (end_mils / 1000000))

    game_gests = gestures.loc[gestures['Inflection'] > relative_start]
    game_gests = game_gests.loc[gestures['Inflection'] < (relative_end + 10)]
    # game_gests =
    game_gests['TrueInflection'] = 0
    for i in range(0, len(game_gests)):
        game_gests.iloc[i, -1] = game_start_time + timedelta(seconds=(game_gests.iloc[i, -7] - relative_start))

    return game_gests


# Reads a .txt log file exported by PokerTH, returns a pandas DataFrame with all game data for use later calculating
# things. Not all the time stamps will have been recorded equally, so the "time stamp buffer" is used to adjust for
# this. Certain types of actions in the logs tend to be more likely to need time stamp adjustments, so most of these
# can be made automatically by this parsing script. A handful may have to be fixed manually, however.
def get_poker_actions(log_filename: str, timestamps: list[str], hero_name: str, villain_name: str) -> pd.DataFrame:
    # Initializing the Data Frame
    blank_data = np.zeros(13).reshape((1, 13))
    poker_actions = pd.DataFrame(blank_data, columns=['HandNum', 'Action Num', 'TimeStamp', 'Small Blind', 'Big Blind',
                                                      'StartHeroTurn', 'Game Phase', 'Hero Chips', 'Villain Chips',
                                                      'Pot Chips', 'Hole Cards', 'Board Cards', 'Log Line'])

    #     print(len(timestamps_))
    # Initializing the starting values for the data frame
    game_num = 1  # Every time a game starts over
    hand_num = 0  # Every time a new hand is dealt
    action_num = -1  #
    hand_start_index = action_num  # This is so that we can keep track of when we knew the hole cards
    shows_action_buffer = 0
    time_stamp_buffer = 0
    time_stamp = 0.0
    tie_detector = False
    small_blind = 10
    big_blind = 20
    hero_went_all_in = False
    villain_went_all_in = False
    hero_turn = False
    prev_show = False
    double_show = False
    game_phase = "STARTUP"
    game_end = False
    hero_chips = 5000
    villain_chips = 5000
    hole_cards = "[?s,?h]"
    board_cards = ""
    pot_chips = 0
    max_commit = 0
    hero_commit = 0
    villain_commit = 0
    hero_name = hero_name
    villain_name = villain_name
    dealer = villain_name
    log_line = ""

    foo = 0
    with open(log_filename) as f:
        time_index = 0
        for line in f:
            log_line = line
            # Ignore the first line, and ignore blank lines
            if "Log-File for PokerTH" in line:
                pass
            elif line in ['\n', '\r\n']:
                pass

            else:
                if "-----------" in line:  # This is the start of a new game, so many defaults should be reset.
                    hand_num += 1
                    words = line.split()
                    if words[5] == "1":
                        hero_chips = 5000
                        villain_chips = 5000
                        game_end = False
                    pot_chips = 0
                    hero_commit = 0
                    villain_commit = 0
                    max_commit = 0
                    hole_cards = "[?s,?h]"
                    board_cards = ""
                    game_phase = "PREFLOP"
                if "BLIND LEVEL" in line:
                    words = line.split()
                    small_blind = int(words[2][1:])
                    big_blind = int(words[4][1:])
                if "BLINDS" in line:
                    max_commit += big_blind

                if villain_name + " starts as dealer" in line:
                    dealer = villain_name
                    if small_blind < villain_chips:
                        villain_chips = villain_chips - small_blind
                        villain_commit += small_blind
                    else:
                        villain_commit += villain_chips
                        villain_chips = 0
                    if big_blind < hero_chips:
                        hero_chips = hero_chips - big_blind
                        hero_commit += big_blind
                    else:
                        hero_commit += hero_chips
                        hero_chips = 0

                    pot_chips = pot_chips + small_blind + big_blind
                if hero_name + " starts as dealer" in line:
                    dealer = hero_name

                    if small_blind < hero_chips:
                        hero_chips = hero_chips - small_blind
                        hero_commit += small_blind
                    else:
                        hero_commit += hero_chips
                        hero_chips = 0
                    if big_blind < villain_chips:
                        villain_chips = villain_chips - big_blind
                        villain_commit += big_blind
                    else:
                        villain_commit += villain_chips
                        villain_chips = 0

                    pot_chips = pot_chips + villain_commit + hero_commit

                if "Seat 1" in line:
                    pass
                    # print("Hero's seat...")
                if "Seat 2" in line:
                    pass
                    # print("Villain's seat...")
                if "PREFLOP" in line:
                    prev_show = False
                    double_show = False
                    tie_detector = False
                    hero_went_all_in = False
                    villain_went_all_in = False
                    if dealer == villain_name:
                        hero_turn = False
                    elif dealer == hero_name:
                        hero_turn = True
                    action_num += 1
                    hand_start_index = action_num
                    # Check the time difference between this action and the next
                    timestamp1 = timestamps[action_num + time_stamp_buffer]
                    timestamp2 = timestamps[action_num + time_stamp_buffer + 1]
                    t1 = datetime.strptime(timestamp1, "%H:%M:%S,%f")
                    t2 = datetime.strptime(timestamp2, "%H:%M:%S,%f")
                    difference = t2 - t1
                    # Sometimes PREFLOPs move through the system too quickly, and timestamps must be adjusted
                    # accordingly
                    if difference.seconds == 0 and difference.microseconds < 100000:
                        print("Found a PREFLOP timestamp difference of " + repr(difference) + " at Hand " + repr(
                            hand_num))
                        print(t1)
                        print(t2)
                        time_stamp_buffer = time_stamp_buffer + 1
                    # Prepare the next row in the DF
                    time_stamp = (timestamps[action_num + time_stamp_buffer])
                    next_action = np.array(
                        [hand_num, action_num, time_stamp, small_blind, big_blind, hero_turn, game_phase, hero_chips,
                         villain_chips, pot_chips, hole_cards, board_cards, log_line]).reshape((1, 13))
                    next_action_df = pd.DataFrame(next_action,
                                                  columns=['HandNum', 'Action Num', 'TimeStamp', 'Small Blind',
                                                           'Big Blind', 'StartHeroTurn', 'Game Phase', 'Hero Chips',
                                                           'Villain Chips', 'Pot Chips', 'Hole Cards', 'Board Cards',
                                                           'Log Line'])
                    new_list = poker_actions, next_action_df
                    poker_actions = pd.concat(new_list)

                if "FLOP [" in line:
                    game_phase = "FLOP"
                    if dealer == villain_name:
                        hero_turn = True
                    elif dealer == hero_name:
                        hero_turn = False
                    words = line.split()
                    board_cards = "[" + words[3]
                    action_num += 1
                    # As above with PREFLOP time differences.
                    timestamp1 = timestamps[action_num + time_stamp_buffer - 1]
                    timestamp2 = timestamps[action_num + time_stamp_buffer]
                    t1 = datetime.strptime(timestamp1, "%H:%M:%S,%f")
                    t2 = datetime.strptime(timestamp2, "%H:%M:%S,%f")
                    difference = t2 - t1
                    if difference.seconds > 1:
                        time_stamp_buffer = time_stamp_buffer - 1

                    # Prepping the next row in the dataframe
                    time_stamp = (timestamps[action_num + time_stamp_buffer])
                    next_action = np.array(
                        [hand_num, action_num, time_stamp, small_blind, big_blind, hero_turn, game_phase, hero_chips,
                         villain_chips, pot_chips, hole_cards, board_cards, log_line]).reshape((1, 13))
                    next_action_df = pd.DataFrame(next_action,
                                                  columns=['HandNum', 'Action Num', 'TimeStamp', 'Small Blind',
                                                           'Big Blind', 'StartHeroTurn', 'Game Phase', 'Hero Chips',
                                                           'Villain Chips', 'Pot Chips', 'Hole Cards', 'Board Cards',
                                                           'Log Line'])
                    new_list = poker_actions, next_action_df
                    poker_actions = pd.concat(new_list)

                if "TURN" in line:
                    game_phase = "TURN"
                    if dealer == villain_name:
                        hero_turn = True
                    elif dealer == hero_name:
                        hero_turn = False
                    words = line.split()
                    board_cards = "[" + words[3]
                    action_num += 1
                    # As above with PREFLOP and FLOP, to detect a missed timestamp
                    timestamp1 = timestamps[action_num + time_stamp_buffer - 1]
                    timestamp2 = timestamps[action_num + time_stamp_buffer]
                    t1 = datetime.strptime(timestamp1, "%H:%M:%S,%f")
                    t2 = datetime.strptime(timestamp2, "%H:%M:%S,%f")
                    difference = t2 - t1
                    if difference.seconds > 1:
                        time_stamp_buffer = time_stamp_buffer - 1

                    # Prepping the next row in the DF
                    time_stamp = (timestamps[action_num + time_stamp_buffer])
                    next_action = np.array(
                        [hand_num, action_num, time_stamp, small_blind, big_blind, hero_turn, game_phase, hero_chips,
                         villain_chips, pot_chips, hole_cards, board_cards, log_line]).reshape((1, 13))
                    next_action_df = pd.DataFrame(next_action,
                                                  columns=['HandNum', 'Action Num', 'TimeStamp', 'Small Blind',
                                                           'Big Blind', 'StartHeroTurn', 'Game Phase', 'Hero Chips',
                                                           'Villain Chips', 'Pot Chips', 'Hole Cards', 'Board Cards',
                                                           'Log Line'])
                    new_list = poker_actions, next_action_df
                    poker_actions = pd.concat(new_list)

                if "RIVER" in line:
                    game_phase = "RIVER"
                    if dealer == villain_name:
                        hero_turn = True
                    elif dealer == hero_name:
                        hero_turn = False
                    words = line.split()
                    board_cards = "[" + words[3]
                    action_num += 1
                    # As above w/PREFLOP, FLOP, and TURN
                    timestamp1 = timestamps[action_num + time_stamp_buffer - 1]
                    timestamp2 = timestamps[action_num + time_stamp_buffer]
                    t1 = datetime.strptime(timestamp1, "%H:%M:%S,%f")
                    t2 = datetime.strptime(timestamp2, "%H:%M:%S,%f")
                    difference = t2 - t1
                    if difference.seconds > 1:
                        time_stamp_buffer = time_stamp_buffer - 1

                    # Prepping the next frame in the DataFrame
                    time_stamp = (timestamps[action_num + time_stamp_buffer])
                    next_action = np.array(
                        [hand_num, action_num, time_stamp, small_blind, big_blind, hero_turn, game_phase, hero_chips,
                         villain_chips, pot_chips, hole_cards, board_cards, log_line]).reshape((1, 13))
                    next_action_df = pd.DataFrame(next_action,
                                                  columns=['HandNum', 'Action Num', 'TimeStamp', 'Small Blind',
                                                           'Big Blind', 'StartHeroTurn', 'Game Phase', 'Hero Chips',
                                                           'Villain Chips', 'Pot Chips', 'Hole Cards', 'Board Cards',
                                                           'Log Line'])
                    new_list = poker_actions, next_action_df
                    poker_actions = pd.concat(new_list)

                if "checks" in line:
                    # No action, really, just update the dataframe.
                    action_num += 1

                    time_stamp = (timestamps[action_num + time_stamp_buffer])
                    hero_turn = not hero_turn
                    next_action = np.array(
                        [hand_num, action_num, time_stamp, small_blind, big_blind, hero_turn, game_phase, hero_chips,
                         villain_chips, pot_chips, hole_cards, board_cards, log_line]).reshape((1, 13))
                    next_action_df = pd.DataFrame(next_action,
                                                  columns=['HandNum', 'Action Num', 'TimeStamp', 'Small Blind',
                                                           'Big Blind', 'StartHeroTurn', 'Game Phase', 'Hero Chips',
                                                           'Villain Chips', 'Pot Chips', 'Hole Cards', 'Board Cards',
                                                           'Log Line'])
                    new_list = poker_actions, next_action_df
                    poker_actions = pd.concat(new_list)
                if "bets" in line:

                    action_num += 1
                    time_stamp = (timestamps[action_num + time_stamp_buffer])
                    hero_turn = not hero_turn
                    if villain_name in line:
                        words = line.split()
                        bet_size = int(words[2][1:-1])
                        villain_chips = villain_chips - bet_size
                        pot_chips += bet_size
                        villain_commit += bet_size
                        # Villain put hero all in
                        if hero_commit < villain_commit:
                            max_commit = villain_commit
                        else:
                            max_commit = hero_commit

                    if hero_name in line:
                        words = line.split()
                        bet_size = int(words[2][1:-1])
                        # print(bet_size)
                        hero_chips = hero_chips - bet_size
                        pot_chips += bet_size
                        # cost_to_stay = max_commit - bet_size
                        hero_commit += bet_size
                        if hero_commit > villain_commit:
                            max_commit = hero_commit
                        else:
                            max_commit = villain_commit

                    # Prep the next row of the data frame
                    next_action = np.array(
                        [hand_num, action_num, time_stamp, small_blind, big_blind, hero_turn, game_phase, hero_chips,
                         villain_chips, pot_chips, hole_cards, board_cards, log_line]).reshape((1, 13))
                    next_action_df = pd.DataFrame(next_action,
                                                  columns=['HandNum', 'Action Num', 'TimeStamp', 'Small Blind',
                                                           'Big Blind', 'StartHeroTurn', 'Game Phase', 'Hero Chips',
                                                           'Villain Chips', 'Pot Chips', 'Hole Cards', 'Board Cards',
                                                           'Log Line'])
                    new_list = poker_actions, next_action_df
                    poker_actions = pd.concat(new_list)
                if "all in" in line:
                    action_num += 1
                    timestamp1 = timestamps[action_num + time_stamp_buffer]
                    timestamp2 = timestamps[action_num + time_stamp_buffer + 3]
                    t1 = datetime.strptime(timestamp1, "%H:%M:%S,%f")
                    t2 = datetime.strptime(timestamp2, "%H:%M:%S,%f")
                    difference = t2 - t1

                    time_stamp = (timestamps[action_num + time_stamp_buffer])
                    hero_turn = not hero_turn
                    if villain_name in line:
                        villain_went_all_in = True
                        words = line.split()
                        if villain_chips > hero_chips and villain_commit >= hero_commit:
                            pot_chips += hero_chips
                            villain_commit += hero_chips
                            max_commit += hero_chips
                            villain_chips -= hero_chips
                        else:
                            pot_chips += villain_chips
                            villain_commit += villain_chips
                            max_commit += villain_chips
                            villain_chips = 0
                    if hero_name in line:
                        hero_went_all_in = True
                        words = line.split()
                        if hero_chips > villain_chips and hero_commit >= villain_commit:
                            pot_chips += villain_chips
                            hero_commit += villain_chips
                            max_commit += villain_chips
                            hero_chips -= villain_chips
                        else:
                            pot_chips += hero_chips
                            hero_commit += hero_chips
                            max_commit += hero_chips
                            hero_chips = 0
                    next_action = np.array(
                        [hand_num, action_num, time_stamp, small_blind, big_blind, hero_turn, game_phase, hero_chips,
                         villain_chips, pot_chips, hole_cards, board_cards, log_line]).reshape((1, 13))
                    next_action_df = pd.DataFrame(next_action,
                                                  columns=['HandNum', 'Action Num', 'TimeStamp', 'Small Blind',
                                                           'Big Blind', 'StartHeroTurn', 'Game Phase', 'Hero Chips',
                                                           'Villain Chips', 'Pot Chips', 'Hole Cards', 'Board Cards',
                                                           'Log Line'])
                    new_list = poker_actions, next_action_df
                    poker_actions = pd.concat(new_list)
                    if difference.seconds > 0:
                        print("Made a timestamp adjustment BECAUSE WE'RE ALL IN")
                        print("At " + repr(game_phase) + " " + ", Hand" + repr(hand_num))

                if "calls" in line:
                    action_num += 1
                    time_stamp = (timestamps[action_num + time_stamp_buffer])
                    hero_turn = not hero_turn
                    if villain_name in line:
                        words = line.split()
                        cost_to_stay = hero_commit - villain_commit
                        pot_chips += cost_to_stay
                        villain_chips -= cost_to_stay
                        villain_commit = max_commit
                    if hero_name in line:
                        words = line.split()
                        cost_to_stay = villain_commit - hero_commit
                        pot_chips += cost_to_stay
                        hero_chips -= cost_to_stay
                        hero_commit = max_commit
                    next_action = np.array(
                        [hand_num, action_num, time_stamp, small_blind, big_blind, hero_turn, game_phase, hero_chips,
                         villain_chips, pot_chips, hole_cards, board_cards, log_line]).reshape((1, 13))
                    next_action_df = pd.DataFrame(next_action,
                                                  columns=['HandNum', 'Action Num', 'TimeStamp', 'Small Blind',
                                                           'Big Blind', 'StartHeroTurn', 'Game Phase', 'Hero Chips',
                                                           'Villain Chips', 'Pot Chips', 'Hole Cards', 'Board Cards',
                                                           'Log Line'])
                    new_list = poker_actions, next_action_df
                    poker_actions = pd.concat(new_list)
                if "shows" in line:  # Someone has just won, so
                    shows_action_buffer += 1
                    # Checking to see if someone shows "unnecessarily", or shows after the chips have already been
                    # distributed
                    if pot_chips == 0 and villain_chips > 0 and hero_chips > 0:
                        # and (game_phase != "RIVER"): #SHOWING AFTER THE
                        shows_action_buffer = shows_action_buffer - 1
                        action_num += 1
                        time_stamp = (timestamps[action_num + time_stamp_buffer])

                    if hero_name in line:
                        words = line.split()
                        if game_phase == "RIVER" and len(words) > 3:
                            hole_cards = "[" + words[3]
                        else:
                            hole_cards = words[2]
                    # This is detecting if both were showing earlier – to detect if both players went all in
                    if prev_show is True:
                        shows_action_buffer += 1
                        prev_show = False
                    hero_turn = False
                    next_action = np.array(
                        [hand_num, action_num, time_stamp, small_blind, big_blind, hero_turn, game_phase, hero_chips,
                         villain_chips, pot_chips, hole_cards, board_cards, log_line]).reshape((1, 13))
                    next_action_df = pd.DataFrame(next_action,
                                                  columns=['HandNum', 'Action Num', 'TimeStamp', 'Small Blind',
                                                           'Big Blind', 'StartHeroTurn', 'Game Phase', 'Hero Chips',
                                                           'Villain Chips', 'Pot Chips', 'Hole Cards', 'Board Cards',
                                                           'Log Line'])
                    new_list = poker_actions, next_action_df
                    poker_actions = pd.concat(new_list)
                    poker_actions.loc[poker_actions['HandNum'] == str(hand_num), 'Hole Cards'] = hole_cards
                    prev_show = True
                if "wins" in line and "!" not in line:
                    if tie_detector is False:
                        action_num += 1

                        # Fixing timestamp issues
                        timestamp1 = timestamps[action_num + time_stamp_buffer]
                        timestamp2 = timestamps[action_num + time_stamp_buffer + 1]
                        timestamp3 = timestamps[action_num + time_stamp_buffer - 1]
                        t1 = datetime.strptime(timestamp1, "%H:%M:%S,%f")
                        t2 = datetime.strptime(timestamp2, "%H:%M:%S,%f")
                        t3 = datetime.strptime(timestamp3, "%H:%M:%S,%f")
                        difference = t2 - t1
                        prev_diff = t1 - t3
                        if difference.seconds == 0 and difference.microseconds < 100000:
                            print("Found a 'wins' timestamp difference of " + repr(difference) + " at Hand " + repr(
                                hand_num))
                            print(t1)
                            print(t2)
                            time_stamp_buffer = time_stamp_buffer + 1
                        if prev_diff.seconds > 1:
                            time_stamp_buffer = time_stamp_buffer - 1
                    time_stamp = (timestamps[action_num + time_stamp_buffer])
                    hero_turn = False
                    # The end of a game
                    if ("game " + str(game_num) + "!") in line:
                        print("GAME OVER")
                        pass
                    elif villain_name in line:
                        words = line.split()
                        # Give the chips to the villain
                        villain_chips += int(words[2][1:])
                        # Fix timestamp differences
                        if villain_went_all_in == True and difference.seconds == 0 and difference.microseconds < 100000:
                            print("moved it forward 1")
                            time_stamp_buffer = time_stamp_buffer + 1
                        villain_went_all_in = False
                        pot_chips = 0
                    elif hero_name in line:
                        words = line.split()
                        # Fix timestamp differences
                        if hero_went_all_in == True and difference.seconds == 0 and difference.microseconds < 100000:
                            print("moved it forward 1")
                            time_stamp_buffer = time_stamp_buffer + 1
                        hero_went_all_in = False
                        # Give the chips to the Hero
                        hero_chips += int(words[2][1:])
                        pot_chips = 0
                    next_action = np.array(
                        [hand_num, action_num, time_stamp, small_blind, big_blind, hero_turn, game_phase, hero_chips,
                         villain_chips, pot_chips, hole_cards, board_cards, log_line]).reshape((1, 13))
                    next_action_df = pd.DataFrame(next_action,
                                                  columns=['HandNum', 'Action Num', 'TimeStamp', 'Small Blind',
                                                           'Big Blind', 'StartHeroTurn', 'Game Phase', 'Hero Chips',
                                                           'Villain Chips', 'Pot Chips', 'Hole Cards', 'Board Cards',
                                                           'Log Line'])
                    new_list = poker_actions, next_action_df
                    poker_actions = pd.concat(new_list)
                    tie_detector = True
                if "folds" in line:
                    action_num += 1
                    time_stamp = (timestamps[action_num + time_stamp_buffer])
                    hero_turn = not hero_turn
                    next_action = np.array(
                        [hand_num, action_num, time_stamp, small_blind, big_blind, hero_turn, game_phase, hero_chips,
                         villain_chips, pot_chips, hole_cards, board_cards, log_line]).reshape((1, 13))
                    next_action_df = pd.DataFrame(next_action,
                                                  columns=['HandNum', 'Action Num', 'TimeStamp', 'Small Blind',
                                                           'Big Blind', 'StartHeroTurn', 'Game Phase', 'Hero Chips',
                                                           'Villain Chips', 'Pot Chips', 'Hole Cards', 'Board Cards',
                                                           'Log Line'])
                    new_list = poker_actions, next_action_df
                    poker_actions = pd.concat(new_list)
                # Ignore this line, but it has probably been time-stamp-logged, so advance the time stamp buffer
                if "sits out" in line:
                    time_stamp_buffer += 1

    print(action_num + time_stamp_buffer)
    return poker_actions
