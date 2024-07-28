import math

from Team import Team
import random


class League:
    name = ""
    teams = []
    teams_hold = []
    matchups = []
    max_games = None
    max_against = None

    def __init__(self, name):
        self.name = name

    def set_max(self,max):
        self.max_games = max
        self.max_against = math.ceil(max/(len(self.teams)-1))

    def add_team(self, name):
        team = Team(name, len(self.teams))
        self.teams.append(team)

    def season(self, weeks):
        self.teams_hold = self.teams.copy()
        for i in range(weeks):
            self.gen_week(self.teams_hold)

    def create_hold(self):
        self.teams_hold = self.teams.copy()
    def matchup_init(self):

        for i in range(len(self.teams)):
            self.matchups.append([])
            '''j=0
            while j < i:
                self.matchups[i].append(0)
                j += 1'''
            for j in range(len(self.teams)):
                self.matchups[i].append(0)
        for i in range(len(self.teams)):
            print(self.matchups[i])

    def gen_week(self, teams):
        temp = teams.copy()
        week = []

        if len(temp) < 2:
            print("Not enough teams left for matchups")
        while len(temp) > 1:

            a = temp.pop(random.randrange(len(temp)))
            b = random.randint(0, len(temp) - 1)
            while (not self.matchup_check(a, temp[b])) and b > -(len(temp)):
                print(temp[b].name,'is not a valid matchup for ',a.name)
                b -= 1


            if self.matchup_check(a, temp[b]):
                temp[b], temp[-1] = temp[-1], temp[b]
                b = temp.pop()
                print("Team a: ", a.name)
                print("Team b: ", b.name)
                self.matchup(a, b)
                week.append([a.name, b.name])
            else:
                print("No valid matchups found for ", a.name)
                #self.teams_hold.remove(a)



        print(week)
        for i in range(len(self.teams)):
            print(self.matchups[i])


    #When a game has been verified the matchup is documented
    def matchup(self, a, b):
        #indices = self.matchup_index(a, b)
        #self.matchups[indices[0]][indices[1]] += 1
        self.matchups[a.index][b.index]+=1
        self.matchups[b.index][a.index]+=1
        a.played_against(b)
        if a.games_played == self.max_games:
            self.teams_hold.remove(a)

        b.played_against(a)
        if b.games_played == self.max_games:
            self.teams_hold.remove(b)
        #for i in range(len(self.teams)):
            #print(self.matchups[i])

    #Used to find the index of the current two teams on the matchups table
    def matchup_index(self, a, b):
        if a.index > b.index:
            out = [a.index, b.index]
        else:
            out = [b.index, a.index]
        return out

    #Checks if a matchup is valid
    def matchup_check(self, a, b):
        indices = self.matchup_index(a, b)
        if (a.last == b.name or b.last == a.name) or self.matchups[indices[0]][indices[1]] == self.max_against :
            return False
        else:
            return True
