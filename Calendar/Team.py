import random
class Team:
    name = ""
    games_played = 0
    last = None
    opp_list = {}
    index = None

    def __init__(self, name, index):
        self.name = name
        self.index = index
    def played_against(self, team):
        self.games_played += 1
        self.last = team.name
        """if self.opp_list[team.name] == 1:
            self.opp_list.pop(team.name)
        else:
            self.opp_list[team.name] -= 1"""

    '''def game_check(self, team):
        if team.name != self.last: #and team.name in self.opp_list:
            return True
        else:
            return False'''

    '''def gen_opps(self, teams, games):
        for i in teams:
            if i.name == self.name:
                continue
            else:
                self.opp_list.update({i.name: games})'''
