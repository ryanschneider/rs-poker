use std::collections::BTreeMap;
use std::fmt;

use tracing::{debug_span, event, instrument, trace_span, Level};
use uuid::Uuid;

use crate::arena::action::{FailedActionPayload, PlayedActionPayload};
use crate::arena::game_state::Round;
use crate::core::{Card, FlatDeck, Rank, Rankable};

use super::action::{
    Action, AgentAction, AwardPayload, DealStartingHandPayload, ForcedBetPayload, GameStartPayload,
    PlayerSitPayload,
};

use super::historian::Historian;
use super::Agent;
use super::GameState;

/// # Description
///
/// This code is implementing a version of Texas Hold'em poker. It is a
/// simulation of the game that can be played with computer agents. The game
/// progresses through a number of rounds: Starting,
/// Preflop,
/// Flop,
/// Turn,
/// River, and
/// Showdown.
///
/// The simulation creates a deck of cards, shuffles it, and deals cards to the
/// players. The players then take turns making bets, raising or folding until
/// the round is complete. Then, the game moves to the next round, and the
/// process repeats. At the end of the game, the player with the best hand wins.
///
/// The simulation is designed to be used with agents that can make decisions
/// based on the game state. The `HoldemSimulation` struct keeps track of the
/// game state, the deck, and the actions taken in the game.
///
/// The `run` method can be used to run the entire game
///
/// # Behavior
///
/// - Any agent bet that is an over bet will silently turn into an all in. That
///   is to say if an agent has 100 in their stack and bet `100_000_000` that
///   will be accepted and will be equivilant to bet `100`
/// - Any bet that `GameState` rules as being impossible, those that turn into
///   [`rs-poker::arena::errors::GameStateError`] will instead be turned into
///   fold.
/// - It's expected that you have the same number of agents as you have chip
///   stacks in the game state. If players are not active, you can use the
///   `FoldingAgent` as a stand in and set the active bit to false.
pub struct HoldemSimulation {
    pub id: Uuid,
    pub agents: Vec<Box<dyn Agent>>,
    pub game_state: GameState,
    pub deck: FlatDeck,
    pub historians: Vec<Box<dyn Historian>>,
}

impl HoldemSimulation {
    pub fn more_rounds(&self) -> bool {
        !matches!(self.game_state.round, Round::Complete)
    }
    /// Returns the number of poker agents participating in this simulation.
    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    /// Run the simulation all the way to completion. This will mutate the
    /// current state.
    pub fn run(&mut self) {
        let span = debug_span!("run",
            game_state = ?self.game_state,
            deck = ?self.deck);
        let _enter = span.enter();

        while self.more_rounds() {
            self.run_round();
        }
    }

    #[instrument]
    pub fn run_round(&mut self) {
        let span = trace_span!("run_round");
        let _enter = span.enter();

        match self.game_state.round {
            // Dealing the user hand is dealt with as its own round
            // in order to use the per round active bit set
            // for iterating players
            Round::Starting => self.start(),
            Round::Preflop => self.preflop(),
            Round::Flop => self.flop(),
            Round::Turn => self.turn(),
            Round::River => self.river(),
            Round::Showdown => self.showdown(),

            // There's nothing left to do to this.
            Round::Complete => (),
        }
    }

    fn start(&mut self) {
        let span = trace_span!("start");
        let _enter = span.enter();

        // Add an action to record the sb and bb
        // This should allow recreating starting game state
        // together with PlayerSit actions.
        self.record_action(Action::GameStart(GameStartPayload {
            small_blind: self.game_state.small_blind,
            big_blind: self.game_state.big_blind,
        }));

        // We deal the cards before advancing the round
        // This allows us to use the round active bitset
        while self.game_state.current_round_num_active_players() > 0 {
            let idx = self.game_state.to_act_idx();

            // Add an action that records starting stack for each player
            // Starting with to the left of the dealer
            // and ending with dealer button.
            self.record_action(Action::PlayerSit(PlayerSitPayload {
                player_stack: self.game_state.stacks[idx],
                idx,
            }));

            self.deal_player_cards(2);

            // set the active bit on the player to false.
            // This allows us to not deal to players that
            // are sitting out, while also going in the same
            // order of dealing
            self.game_state
                .round_data
                .as_mut()
                .unwrap()
                .player_active
                .disable(idx);

            self.game_state
                .round_data
                .as_mut()
                .unwrap()
                .advance_action();
        }

        // We're done with the non-betting dealing only round
        self.advance_round();
    }

    fn preflop(&mut self) {
        let span = trace_span!("preflop");
        let _enter = span.enter();

        // We have two different bets to force.
        let sb = self.game_state.small_blind;
        let sb_idx = self.game_state.to_act_idx();
        self.game_state.do_bet(sb, true).unwrap();
        self.record_action(Action::ForcedBet(ForcedBetPayload {
            bet: sb,
            idx: sb_idx,
            player_stack: self.game_state.stacks[sb_idx],
        }));

        let bb = self.game_state.big_blind;
        let bb_idx = self.game_state.to_act_idx();
        self.game_state.do_bet(bb, true).unwrap();
        self.record_action(Action::ForcedBet(ForcedBetPayload {
            bet: bb,
            idx: bb_idx,
            player_stack: self.game_state.stacks[bb_idx],
        }));

        self.run_betting_round();
        self.advance_round();
    }

    fn flop(&mut self) {
        let span = trace_span!("flop");
        let _enter = span.enter();

        self.deal_comunity_cards(3);
        self.run_betting_round();
        self.advance_round();
    }

    fn turn(&mut self) {
        let span = trace_span!("turn");
        let _enter = span.enter();

        self.deal_comunity_cards(1);
        self.run_betting_round();
        self.advance_round();
    }

    fn river(&mut self) {
        let span = trace_span!("river");
        let _enter = span.enter();

        self.deal_comunity_cards(1);
        self.run_betting_round();
        self.advance_round();
    }

    fn showdown(&mut self) {
        let span = trace_span!("showdown");
        let _enter = span.enter();

        // Rank each player that still has a chance.
        let active = self.game_state.player_active | self.game_state.player_all_in;

        let mut bets = self.game_state.player_bet.clone();

        // Create a map where the keys are the ranks of hands and
        // the values are vectors of player index, for players that had that hand
        let ranks = active
            .ones()
            .map(|idx| (idx, self.game_state.hands[idx].rank()))
            .fold(
                BTreeMap::new(),
                |mut map: BTreeMap<Rank, Vec<usize>>, (idx, rank)| {
                    map.entry(rank)
                        .and_modify(|m| {
                            m.push(idx);
                            m.sort_by(|a, b| bets[*a].partial_cmp(&bets[*b]).unwrap());
                        })
                        .or_insert_with(|| vec![idx]);

                    map
                },
            );

        // By default the map gives keys in assending order. We want them descending.
        // The actual player vector is sorted in ascending order according to bet size.
        for (rank, players) in ranks.into_iter().rev() {
            let mut start_idx = 0;
            let end_idx = players.len();

            // Set the rank on the current round since we know the ranks
            for idx in players.iter() {
                let round_data = self.game_state.round_data.as_mut().unwrap();
                round_data.set_hand_rank(*idx, rank);
            }

            // We'll conitune until every player has been given the matching money
            // up to their wager. However since some players might have gone allin
            // earlier we keep removing from the pot and splitting it equally to all
            // those players still left in the pot.
            while start_idx < end_idx {
                // Becasue our lists are ordered from smallest bets to largest
                // we can just assume the first one is the smallest
                //
                // Here we use that property to find the max bet that this pot
                // will give for this round of splitting ties.
                let max_wager = bets[players[start_idx]];
                let mut pot: f64 = 0.0;

                // Most common is that ties will
                // be for wagers that are all the same.
                // So check if there's no more
                // bets to award for this player.
                if max_wager <= 0.0 {
                    start_idx += 1;
                    continue;
                }

                // Take all the wagers remaining into a
                // side pot. However this side pot might
                // be the only pot if there were no allins
                for b in bets.iter_mut() {
                    let w = (*b).min(max_wager);
                    *b -= w;
                    pot += w as f64;
                }

                // Now all the winning players get
                // an equal share of the side pot
                let num_players = (end_idx - start_idx) as f64;
                let split = pot / num_players;

                for idx in &players[start_idx..end_idx] {
                    // Record that this player won something
                    event!(parent: &span, Level::INFO, idx, split, pot, ?rank, "pot_awarded");
                    self.game_state.award(*idx, split as f32);
                    self.record_action(Action::Award(AwardPayload {
                        idx: *idx,
                        total_pot: pot as f32,
                        award_ammount: split as f32,
                        // Since we had a showdown we cen copy the hand
                        // and the resulting rank.
                        rank: Some(rank),
                        hand: Some(self.game_state.hands[*idx].clone()),
                    }));
                }

                // Since the first player is bet size
                // that we used. They have won everything that they're eligible for.
                start_idx += 1;
            }
        }

        self.end_game();
    }

    fn deal_player_cards(&mut self, num_cards: usize) {
        let new_hand: Vec<Card> = self.deal_cards(num_cards);
        for c in &new_hand {
            self.record_action(Action::DealStartingHand(DealStartingHandPayload {
                card: *c,
                idx: self.game_state.to_act_idx(),
            }));
        }

        let idx = self.game_state.to_act_idx();
        self.game_state.hands[idx].extend(new_hand);
    }

    fn deal_comunity_cards(&mut self, num_cards: usize) {
        let mut community_cards = self.deal_cards(num_cards);
        for c in &community_cards {
            self.record_action(Action::DealCommunity(*c));
        }
        // Add all the cards to the hands as well.
        for h in &mut self.game_state.hands {
            h.extend(community_cards.to_owned());
        }
        // Drain the community_cards vec into the game_state board.
        self.game_state.board.append(&mut community_cards);
    }

    /// Pull num_cards from the deck and return them as a vector.
    fn deal_cards(&mut self, num_cards: usize) -> Vec<Card> {
        let mut cards: Vec<Card> = (0..num_cards).map(|_| self.deck.deal().unwrap()).collect();

        // Keep the cards sorted in min to max order
        // this keeps the number of permutations down since
        // its now AsKsKd is the same as KdAsKs after sorting.
        cards.sort();

        cards
    }

    /// This runs betting for the round to completion. It will run until
    /// everyone has acted or until the round has been completed because no one
    /// can act anymore.
    fn run_betting_round(&mut self) {
        // There's no need to run any betting round if there's no on left in the round.
        if self.game_state.current_round_num_active_players() > 1 {
            let current_round = self.game_state.round;
            // However if there is more than one player,
            // we need to run the betting until everyone has acted.
            // or until the round has been completed
            // because no one can act anymore
            while self.game_state.current_round_num_active_players() > 0
                && current_round == self.game_state.round
            {
                self.run_single_agent();
            }
        }
    }

    /// Run the next agent in the game state to act.
    fn run_single_agent(&mut self) {
        let idx = self.game_state.to_act_idx();
        let span = trace_span!("run_agent", idx);
        let _enter = span.enter();
        let action = self.agents[idx].act(&self.id, &self.game_state);

        event!(parent: &span, Level::TRACE, ?action, idx);
        self.run_agent_action(action)
    }

    /// Given the action that an agent wants to take, this function will
    /// determine if the action is valid and then apply it to the game state.
    /// If the action is invalid, the agent will be forced to fold.
    fn run_agent_action(&mut self, agent_action: AgentAction) {
        event!(Level::TRACE, ?agent_action, "run_agent_action");

        let idx = self.game_state.to_act_idx();

        match agent_action {
            AgentAction::Fold => {
                let player_bet = self.game_state.current_round_player_bet(idx);
                let current_bet = self.game_state.current_round_bet();

                // For now do exact, but this might be a place we should
                // use approxiate compaison crate. But that's optional only now.
                if player_bet == current_bet {
                    event!(Level::WARN, "fold_error");

                    let new_action = AgentAction::Bet(current_bet);

                    self.game_state.do_bet(current_bet, false).unwrap();
                    self.record_action(Action::FailedAction(FailedActionPayload {
                        action: agent_action,
                        result_action: new_action,
                        player_stack: self.game_state.stacks[idx],
                        idx,
                    }));
                } else {
                    self.record_action(Action::PlayedAction(PlayedActionPayload {
                        action: agent_action,
                        player_stack: self.game_state.stacks[idx],
                        idx,
                    }));
                    self.player_fold();
                }
            }
            AgentAction::Bet(bet_ammount) => {
                let bet_result = self.game_state.do_bet(bet_ammount, false);

                match bet_result {
                    Err(error) => {
                        // If the agent failed to give us a good bet then they lose this round.
                        //
                        // We emit the error as an event
                        // Assume that game_state.do_bet() will have changed nothing since it
                        // errored out Add an action that shows the user was
                        // force folded. Actually fold the user
                        event!(Level::WARN, ?error, "bet_error");

                        // Record this errant action
                        self.record_action(Action::FailedAction(FailedActionPayload {
                            action: agent_action,
                            result_action: AgentAction::Fold,
                            player_stack: self.game_state.stacks[idx],
                            idx,
                        }));

                        // Actually fold the user
                        self.player_fold();
                    }
                    Ok(_added) => {
                        let player_bet = self.game_state.current_round_player_bet(idx);

                        let new_action = AgentAction::Bet(player_bet);
                        // If the game_state.do_bet function returned Ok then
                        // the state is already changed so record the action as played.
                        self.record_action(Action::PlayedAction(PlayedActionPayload {
                            action: new_action,
                            player_stack: self.game_state.stacks[idx],
                            idx,
                        }));
                    }
                }
            }
        }
    }

    #[instrument]
    fn player_fold(&mut self) {
        self.game_state.fold();
        let left = self.game_state.player_active | self.game_state.player_all_in;

        // If there's only one person left then they win.
        // If there's no one left, and one person went all in they win.
        //
        if left.count() <= 1 {
            if let Some(winning_idx) = left.ones().next() {
                let total_pot = self.game_state.total_pot;
                event!(Level::INFO, winning_idx, total_pot, "folded_to_winner");
                self.game_state.award(winning_idx, total_pot);
                self.record_action(Action::Award(AwardPayload {
                    idx: winning_idx,
                    total_pot,
                    award_ammount: total_pot,
                    rank: None,
                    hand: None,
                }))
            }

            self.end_game();
        }
    }

    #[instrument]
    fn end_game(&mut self) {
        let current_round = self.game_state.round;
        self.game_state.complete();
        if current_round != self.game_state.round {
            self.record_action(Action::RoundAdvance(self.game_state.round));
        }
    }

    #[instrument]
    fn advance_round(&mut self) {
        let current_round = self.game_state.round;
        self.game_state.advance_round();
        if self.game_state.round != current_round {
            self.record_action(Action::RoundAdvance(self.game_state.round));
        }
    }

    fn record_action(&mut self, action: Action) {
        event!(Level::TRACE, action = ?action, game_state = ?self.game_state, "add_action");
        for historian in &mut self.historians {
            historian.record_action(&self.id, &self.game_state, action.clone())
        }
    }
}

impl fmt::Debug for HoldemSimulation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HoldemSimulation")
            .field("game_state", &self.game_state)
            .field("deck", &self.deck)
            .finish()
    }
}
