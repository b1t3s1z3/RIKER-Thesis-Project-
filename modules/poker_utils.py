import numpy as np
import pandas as pd
import holdem_calc
from datetime import datetime


def get_fg(game_acts: pd.DataFrame, subject_num: int) -> pd.DataFrame:
    fg = pd.read_csv(f'data/processed/valid_fg_{subject_num}.csv')
    fg = fg.drop('Unnamed: 0', 1)
    start_time = datetime.strptime(game_acts.iloc[0, 2], "%H:%M:%S,%f")
    fg['RelativeTime'] = 0
    fg['RelativeSec'] = 0

    for i in range(0, len(fg)):
        if len(fg.iloc[i, -3]) > 19:
            cur_time = datetime.strptime(fg.iloc[i, -3][11:], "%H:%M:%S.%f")
        else:
            cur_time = datetime.strptime(fg.iloc[i, -3][11:], "%H:%M:%S")

        difference = cur_time - start_time
        fg.iloc[i, -2] = difference
        fg.iloc[i, -1] = difference.seconds + (difference.microseconds / 1000000)  # + (difference.minutes*60)

    return fg


def convert_preflop_hand(preflop_hand: pd.Series) -> str:
    first_suit = ""
    second_suit = ""
    first_val = ""
    second_val = ""
    first_key = ""
    second_key = ""
    s_or_o = ""

    faces = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

    for i, c in enumerate(preflop_hand):
        if i == 1:
            first_val = c
        if i == 2:
            first_suit = c
        if i == 4:
            second_val = c
        if i == 5:
            second_suit = c
    just_cards = first_val + first_suit + second_val + second_suit

    if first_suit == second_suit:
        s_or_o = "s"
    else:
        s_or_o = "o"
    if first_val in faces:
        first_key = faces[first_val]
    else:
        first_key = int(first_val)
    if second_val in faces:
        second_key = faces[second_val]
    else:
        second_key = int(second_val)

    vals = [first_key, second_key]
    vals.sort(key=int, reverse=True)
    if vals[0] > 9:
        alpha_key = (list(faces.keys())[list(faces.values()).index(vals[0])])
    else:
        alpha_key = vals[0]
    if vals[1] > 9:
        beta_key = (list(faces.keys())[list(faces.values()).index(vals[1])])
    else:
        beta_key = vals[1]
    if alpha_key in faces:
        first_val = alpha_key
    else:
        first_val = vals[0]
    if beta_key in faces:
        second_val = beta_key
    else:
        second_val = vals[1]

    preflop_hand = str(first_val) + str(second_val) + s_or_o

    return preflop_hand


def get_preflop_win_analysis(poker_actions: pd.DataFrame) -> pd.DataFrame:
    preflop_stats = pd.DataFrame.from_csv('data/raw/preflop_odds.tsv', sep='\t', header=0)
    poker_actions['Win%'] = "?"

    for i in range(0, len(poker_actions)):
        if poker_actions.iloc[i, 6] == "PREFLOP" and poker_actions.iloc[i, 10] != "[?s,?h]":
            preflop_hand = convert_preflop_hand(poker_actions.iloc[i, 10])
            win_odds = preflop_stats.loc[preflop_stats['Name'] == preflop_hand, 'Win %'].iloc[0]
            poker_actions.iat[i, 13] = (win_odds / 100)

    return poker_actions


def postflop_win_analysis(poker_actions: pd.DataFrame) -> pd.DataFrame:
    previous_phase = "PREFLOP"
    hand_win_odds = ""
    already_calculated = False
    for i in range(0, len(poker_actions)):
        current_phase = poker_actions.iloc[i, 6]
        if poker_actions.iloc[i, 6] != "PREFLOP" and already_calculated == True:
            poker_actions.iloc[i, 13] = hand_win_odds

        if current_phase != previous_phase:
            already_calculated = False
            if poker_actions.iloc[i, 6] != "PREFLOP" and poker_actions.iloc[i, 10] != "[?s,?h]":
                print("GETTING NEW WIN % at phase " + repr(current_phase))
                raw_board_cards = poker_actions.iloc[i, 11]

                board_cards = []
                if len(raw_board_cards) == 10:
                    board_cards = [str(raw_board_cards[1:3]), str(raw_board_cards[4:6]), str(raw_board_cards[7:9])]
                if len(raw_board_cards) == 13:
                    board_cards = [str(raw_board_cards[1:3]), str(raw_board_cards[4:6]), str(raw_board_cards[7:9]),
                                   str(raw_board_cards[10:12])]
                if len(raw_board_cards) == 16:
                    board_cards = [str(raw_board_cards[1:3]), str(raw_board_cards[4:6]), str(raw_board_cards[7:9]),
                                   str(raw_board_cards[10:12]), str(raw_board_cards[13:15])]

                holes = poker_actions.iloc[i, 10]
                hole_one = str(holes[1:3])
                hole_two = str(holes[4:6])
                win_odds = holdem_calc.calculate(board_cards, True, 1, None, [hole_one, hole_two, "?", "?"], False)
                print(win_odds[1])
                hand_win_odds = win_odds[1]
                poker_actions.iloc[i, 13] = hand_win_odds
                already_calculated = True

        previous_phase = current_phase

    return poker_actions


def win_odds(poker_acts: pd.DataFrame) -> pd.DataFrame:
    poker_wins = get_preflop_win_analysis(poker_acts)
    poker_wins = postflop_win_analysis(poker_wins)

    return poker_wins


def get_poker_acts(subject_num: int) -> pd.DataFrame:
    poker_acts = pd.read_csv(f'data/processed/valid_poker_acts_{subject_num}.csv')
    poker_acts['Hole Cards'] = poker_acts['Hole Cards'].fillna('[?s,?h]')
    poker_acts['Board Cards'] = poker_acts['Board Cards'].fillna('')
    poker_acts = win_odds(poker_acts)
    start_time = datetime.strptime(poker_acts.iloc[0, 2], "%H:%M:%S,%f")

    poker_acts['RelativeTime'] = 0
    poker_acts['RelativeSec'] = 0

    for i in range(0, len(poker_acts)):
        cur_time = datetime.strptime(poker_acts.iloc[i, 2], "%H:%M:%S,%f")

        difference = cur_time - start_time
        poker_acts.iloc[i, -2] = difference
        poker_acts.iloc[i, -1] = difference.seconds + (difference.microseconds / 1000000)

    return poker_acts


def calc_poker_stats(game_acts: pd.Series, hero_name: str, villain_name: str) -> pd.Series:
    game_acts['M Ratio'] = '?'
    game_acts['EV'] = '?'
    game_acts['Pot Odds'] = '?'
    game_acts['Pot Odds CV'] = '?'
    game_acts['Implied Odds'] = '?'
    game_acts['Fold %'] = '?'
    game_acts['Fold Equity'] = '?'
    game_acts['Semi-Bluff Fold %'] = '?'
    game_acts['Time to Act'] = '?'
    game_acts['Relative Bet'] = '?'
    game_acts['Call Stakes'] = '?'

    for i in range(0, len(game_acts)):
        if i > 0:
            game_acts.loc[i, 'Time to Act'] = game_acts.loc[i, 'RelativeSec'] - game_acts.loc[i - 1, 'RelativeSec']
            current_line = game_acts.loc[i, 'Log Line']
            current_words = current_line.split()
            prev_line = game_acts.loc[i - 1, 'Log Line']
            words = prev_line.split()
            game_acts.loc[i, 'M Ratio'] = min(
                game_acts.loc[i, 'Hero Chips'],
                game_acts.loc[i, 'Villain Chips']) / (
                                                  game_acts.loc[i, 'Small Blind'] +
                                                  game_acts.loc[i, 'Big Blind'])
            if hero_name in current_words and ("calls" in current_words or "bets" in current_words):
                call_amt = int(current_words[2][1:].strip('.'))
                game_acts.loc[i, 'Pot Odds'] = call_amt / (game_acts.loc[i, 'Pot Chips'])
                game_acts.loc[i, 'Relative Bet'] = (call_amt / game_acts.loc[
                    i, 'M Ratio'])  # (PokerActs_from_game_.loc[i,'Hero Chips'] + call_amt))
            if hero_name in current_words and ("bets" in current_words or (
                    "all in with" in current_words and ((villain_name + " is all in with") not in words))):
                bet_amt = int(current_words[2][1:].strip('.'))
                game_acts.loc[i, 'Fold %'] = bet_amt / (game_acts.loc[i, 'Pot Chips'])

            # Calculating EV, Call Stakes, Pot Odds, and Implied Odds, and Semi-Bluff FOld%
            if game_acts.loc[i, 'Win%'] != '?':

                if villain_name in words and "bets" in words:
                    facing_bet = (int(words[2][1:].strip('.')))
                    game_acts.loc[i, 'EV'] = (game_acts.loc[i, 'Win%'] * game_acts.loc[
                        i, 'Pot Chips']) - (1 - game_acts.loc[i, 'Win%']) * facing_bet
                if villain_name in current_words and "bets" in current_words:
                    facing_bet = (int(current_words[2][1:].strip('.')))
                    game_acts.loc[i, 'Call Stakes'] = facing_bet / game_acts.loc[i, 'M Ratio']
                elif "PREFLOP" in current_words:
                    facing_bet = game_acts.loc[i, 'Small Blind']
                    game_acts.loc[i, 'EV'] = (game_acts.loc[i, 'Win%'] * game_acts.loc[
                        i, 'Pot Chips']) - (1 - game_acts.loc[i, 'Win%']) * facing_bet
                elif "FLOP" in current_words or "TURN" in current_words or "RIVER" in current_words:
                    facing_bet = 0
                    game_acts.loc[i, 'EV'] = (game_acts.loc[i, 'Win%'] * game_acts.loc[
                        i, 'Pot Chips']) - (1 - game_acts.loc[i, 'Win%']) * facing_bet
                if hero_name in current_words and ("calls" in current_words or "bets" in current_words):

                    call_amt = int(current_words[2][1:].strip('.'))
                    game_acts.loc[i, 'Pot Odds CV'] = float(game_acts.loc[i, 'Win%']) - float(
                        game_acts.loc[i, 'Pot Odds'])
                    if game_acts.loc[i, 'Win%'] != 0:
                        game_acts.loc[i, 'Implied Odds'] = (call_amt / float(
                            game_acts.loc[i, 'Win%'])) - game_acts.loc[i, 'Pot Chips']

                if hero_name in current_words and ("bets" in current_words or (
                        "all in with" in current_words and ((villain_name + " is all in with") not in words))):
                    game_acts.loc[i, 'Semi-Bluff Fold %'] = float(game_acts.loc[i, 'Fold %']) - (
                            1.5 * (game_acts.loc[i, 'Win%']))

    return game_acts


def name_poker_events(game_acts: pd.Series, hero_name: str, villain_name: str) -> pd.Series:
    game_acts['Win/Lose Events'] = 'Not Applicable'  # 'Toss Up'
    game_acts['Win/Loss'] = 'Not Applicable'
    game_acts['New Card Events'] = 'Not Applicable'  # 'No News'
    game_acts['Calling Events'] = 'Not Applicable'  # 'Toss Up Call'
    game_acts['Betting Events'] = 'Not Applicable'  # 'Toss Up Bet'
    game_acts['Bet/Bluff'] = 'Not Applicable'
    game_acts['C/R/F'] = 'Not Applicable'

    for i in range(0, len(game_acts)):
        if game_acts.loc[i, 'Win%'] != '?':
            current_line = game_acts.loc[i, 'Log Line']
            current_words = current_line.split()
            #             print("NOW POKER EVENTS")
            #             print(current_words)

            #                 previous_line = PokerActs_from_game_.loc[i-1,'Log Line']
            #                 previous_words = previous_line.split()
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) > 0.75:
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Favored Win'
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) > 0.85:
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Very Favored Win'
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) < 0.35:
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Lucky Win'
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) < 0.15:
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Very Lucky Win'
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) < 0.35 and villain_name_ in previous_words and "folds" in previous_words:
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Villain Bluffed Out'
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) > 0.75 and villain_name_ in previous_words and "folds" in previous_words:
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Villain Forced Out'
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) >= 0.35 and float(PokerActs_from_game_.loc[i,'Win%'] <= 0.75):
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Toss Up Win'
            # Losses

            #                 previous_line = PokerActs_from_game_.loc[i-1,'Log Line']
            #                 previous_words = previous_line.split()
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) > 0.75:
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Unlucky Loss'
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) > 0.85:
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Very Unlucky Loss'
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) < 0.35:
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Predictable Loss'
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) < 0.15:
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Very Predictable Loss'
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) < 0.35 and hero_name_ in previous_words and "folds" in previous_words:
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Hero Discretely Bows Out'
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) > 0.75 and hero_name_ in previous_words and "folds" in previous_words:
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Hero Cedes Strong Position'
            #                 if float(PokerActs_from_game_.loc[i,'Win%']) >= 0.35 and float(PokerActs_from_game_.loc[i,'Win%'] <= 0.75):
            #                     PokerActs_from_game_.at[i,'Win/Lose Events'] = 'Toss Up Loss'

            # Preflop Reveals
            if "PREFLOP" in current_words:
                if float(game_acts.loc[i, 'Win%']) > .60:
                    game_acts.at[i, 'New Card Events'] = 'Good Deal'
                if float(game_acts.loc[i, 'Win%']) < .35:
                    game_acts.at[i, 'New Card Events'] = 'Lousy Deal'
                if float(game_acts.loc[i, 'Win%']) >= 0.35 and float(
                        game_acts.loc[i, 'Win%'] <= 0.60):
                    game_acts.at[i, 'New Card Events'] = 'Toss Up Deal'

            # Flop/Turn/River Reveals
            if "FLOP" in current_words or "TURN" in current_words or "RIVER" in current_words:
                #                 if (float(PokerActs_from_game_.loc[i,'Win%']) - float(PokerActs_from_game_.loc[i-1,'Win%'])) > .25:
                #                     PokerActs_from_game_.at[i,'New Card Events'] = 'Great Break'
                if (float(game_acts.loc[i, 'Win%']) - float(game_acts.loc[i - 1, 'Win%'])) > .15:
                    game_acts.at[i, 'New Card Events'] = 'Lucky Break'
                elif (float(game_acts.loc[i, 'Win%']) - float(game_acts.loc[i - 1, 'Win%'])) > 0:
                    game_acts.at[i, 'New Card Events'] = 'Positive Break'
                #                 elif (float(PokerActs_from_game_.loc[i,'Win%']) - float(PokerActs_from_game_.loc[i-1,'Win%'])) < -.25:
                #                     PokerActs_from_game_.at[i,'New Card Events'] = 'Awful Break'
                elif (float(game_acts.loc[i, 'Win%']) - float(
                        game_acts.loc[i - 1, 'Win%'])) < -.15:
                    game_acts.at[i, 'New Card Events'] = 'Unlucky Break'
                elif (float(game_acts.loc[i, 'Win%']) - float(game_acts.loc[i - 1, 'Win%'])) <= 0:
                    game_acts.at[i, 'New Card Events'] = 'Negative Break'
                else:
                    game_acts.at[i, 'New Card Events'] = 'Not Applicable'

            # Calling Events
            if hero_name in current_words and "calls" in current_words and \
                    game_acts.loc[i - 1, 'Call Stakes'] != '?':
                if float(game_acts.loc[i - 1, 'Call Stakes']) <= 2:
                    if float(game_acts.loc[i, 'Win%']) < 0.35:
                        game_acts.at[i, 'Calling Events'] = 'Loose Low-Risk Call'
                    if float(game_acts.loc[i, 'Win%']) > 0.75:
                        game_acts.at[i, 'Calling Events'] = 'Weak Low-Risk Call'
                elif float(game_acts.loc[i - 1, 'Call Stakes']) >= 10 and float(
                        game_acts.loc[i - 1, 'Call Stakes']) <= 50:
                    if float(game_acts.loc[i, 'Win%']) < 0.40:
                        game_acts.at[i, 'Calling Events'] = 'Risky Call'
                    if float(game_acts.loc[i, 'Win%']) > 0.75:
                        game_acts.at[i, 'Calling Events'] = 'Strong Call'
                elif float(game_acts.loc[i - 1, 'Call Stakes']) > 50:
                    if float(game_acts.loc[i, 'Win%']) < 0.50:
                        game_acts.at[i, 'Calling Events'] = 'Bad Call'
                    if float(game_acts.loc[i, 'Win%']) > 0.85:
                        game_acts.at[i, 'Calling Events'] = 'Finishing Call'
                else:
                    game_acts.at[i, 'Calling Events'] = 'Toss Up Call'

            # Betting Events
            if hero_name in current_words and "bets" in current_words and \
                    game_acts.loc[i, 'Relative Bet'] != '?':
                if float(game_acts.loc[i, 'Win%']) < 0.35:
                    game_acts.at[i, 'Bet/Bluff'] = 'Bluff'
                elif float(game_acts.loc[i, 'Win%']) < 0.60:
                    game_acts.at[i, 'Bet/Bluff'] = 'Semi-Bluff'
                else:
                    game_acts.at[i, 'Bet/Bluff'] = 'Bet'

            if hero_name in current_words and "bets" in current_words \
                    and game_acts.loc[i, 'Relative Bet'] != '?':
                if float(game_acts.loc[i, 'Relative Bet']) <= 2:
                    if float(game_acts.loc[i, 'Win%']) < 0.35:
                        game_acts.at[i, 'Betting Events'] = 'Loose Low-Risk Bet'
                    if float(game_acts.loc[i, 'Win%']) < 0.15:
                        game_acts.at[i, 'Betting Events'] = 'Useless Bluff'
                    if float(game_acts.loc[i, 'Win%']) > 0.75:
                        game_acts.at[i, 'Betting Events'] = 'Weak Low-Risk Bet'
                if float(game_acts.loc[i, 'Relative Bet']) >= 10 and float(
                        game_acts.loc[i, 'Relative Bet']) <= 50:
                    if float(game_acts.loc[i, 'Win%']) < 0.50:
                        game_acts.at[i, 'Betting Events'] = 'Risky Bet'
                    if float(game_acts.loc[i, 'Win%']) < 0.35:
                        game_acts.at[i, 'Betting Events'] = 'Risky Bluff'
                    if float(game_acts.loc[i, 'Win%']) > 0.75:
                        game_acts.at[i, 'Betting Events'] = 'Strong Bet'
                if float(game_acts.loc[i, 'Relative Bet']) > 50:
                    if float(game_acts.loc[i, 'Win%']) < 0.60:
                        game_acts.at[i, 'Betting Events'] = 'Gutsy Bet'
                    if float(game_acts.loc[i, 'Win%']) < 0.45:
                        game_acts.at[i, 'Betting Events'] = 'Gutsy Bluff'
                    if float(game_acts.loc[i, 'Win%']) > 0.75:
                        game_acts.at[i, 'Betting Events'] = 'Finishing Bet'
                else:
                    game_acts.at[i, 'Betting Events'] = 'Toss Up Bet'

        # CRF Events
        current_line = game_acts.loc[i, 'Log Line']
        #         print("CURRENT LINE IS")
        #         print(current_line)
        current_words = current_line.split()
        if villain_name in current_words and "bets" in current_words:
            #             print("VILLAIN MADE A BET!")
            #             print("At hand " + repr(PokerActs_from_game_.loc[i,'HandNum']))
            #             print(current_words)
            next_line = game_acts.iloc[i + 1, 12]
            next_words = next_line.split()
            #             print(next_words)
            if hero_name in next_words and "bets" in next_words:
                game_acts.at[i, 'C/R/F'] = 'Raise Response'
            if hero_name in next_words and "folds." in next_words:
                game_acts.at[i, 'C/R/F'] = 'Fold Response'
            if hero_name in next_words and "calls" in next_words:
                game_acts.at[i, 'C/R/F'] = 'Call Response'

        # Wins
        if hero_name in current_words and "wins" in current_words and "!" not in current_words:
            game_acts.at[i, 'Win/Loss'] = 'Win'
        # Losses
        if villain_name in current_words and "wins" in current_words and "!" not in current_words:
            game_acts.at[i, 'Win/Loss'] = 'Loss'

    return game_acts
