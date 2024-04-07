#[macro_use]
extern crate criterion;
extern crate rand;
extern crate rs_poker;

use criterion::Criterion;
use rs_poker::core::{Deck, FlatDeck, Hand, Rankable};

fn rank_all_five(c: &mut Criterion) {
    let deck: FlatDeck = Deck::default().into();

    let mut group = c.benchmark_group("rank all");
    group.measurement_time(std::time::Duration::from_secs(10));
    group.bench_function("Rank all hands", move |b| {
        b.iter(|| {
            for i in 0..48 {
                for j in i + 1..49 {
                    for k in j + 1..50 {
                        for l in k + 1..51 {
                            for m in l + 1..52 {
                                let cards = vec![deck[i], deck[j], deck[k], deck[l], deck[m]];
                                let hand = Hand::new_with_cards(cards);
                                hand.rank_five();
                            }
                        }
                    }
                }
            }
        })
    });
    group.finish()
}

criterion_group!(benches, rank_all_five);
criterion_main!(benches);
