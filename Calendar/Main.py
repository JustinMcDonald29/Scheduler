import League
def main():
    league = League.League("Adults")

    league.add_team("Leafs")
    league.add_team("Bruins")
    league.add_team("Canadiens")
    league.add_team("Red Wings")
    league.add_team("Blackhawks")
    league.add_team("Rangers")
    league.add_team("Senators")
    league.matchup_init()
    league.create_hold()
    league.set_max(12)
    while len(league.teams_hold) > 1 :

        league.gen_week(league.teams_hold)

    for item in league.teams:
        print(item.name,"played ",item.games_played,"games")



main()