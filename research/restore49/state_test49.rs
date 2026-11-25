#[cfg(test)]
mod size_family_contract49 {
    use super::*;
    #[test]
    fn boundary_undo_d4_hash() {
        let c=N/2;
        let seq=[(c,c),(N-1,N-1),(c,c+1),(N-1,0),(c,c-1),(0,N-1),(c,c+2),(0,0),(c,c-2)];
        let mut s=Gomoku15::freestyle();assert_eq!(s.num_actions(),N2);
        let initial=format!("{:?}",s);
        let mut undo=Vec::new();
        let mut snapshots=Vec::new();
        for (i,(y,x)) in seq.into_iter().enumerate(){
            let a=(y*N+x) as u16;
            assert!(s.legal_moves().contains(&a));
            let snap=(format!("{:?}",s),s.encode_planes(),s.tt_hash());
            snapshots.push(snap);
            let pure=s.apply_move(a);
            undo.push(s.apply_move_in_place(a));
            assert_eq!(format!("{:?}",s),format!("{:?}",pure));
            assert_eq!(s.encode_planes().len(),17*N2);
            assert_eq!(s.current_player(),if i%2==0{-1}else{1});
            let b=s.board_state_record().into_iter().map(|x|x as i8).collect::<Vec<_>>();
            let r=Gomoku15::from_board(&b,s.current_player(),GomokuVariant::Freestyle);
            assert_eq!(s.tt_hash(),r.tt_hash());
            assert_eq!(s.is_terminal(),r.is_terminal());
            if i<8 {assert!(!s.is_terminal());} else {assert!(s.is_terminal());assert_eq!(s.outcome(),-1.);}
        }
        while let Some(u)=undo.pop(){
            s.undo_move(u);
            let a=snapshots.pop().unwrap();
            assert_eq!(format!("{:?}",s),a.0);assert_eq!(s.encode_planes(),a.1);assert_eq!(s.tt_hash(),a.2);
        }
        assert_eq!(format!("{:?}",s),initial);
        // Every D4 transform of the same legal terminal history.
        for flip in [false,true] { for rot in 0..4 {
            let mut s=Gomoku15::freestyle();
            for (mut y,mut x) in seq {
                if flip {x=N-1-x;}
                for _ in 0..rot {(y,x)=(x,N-1-y);}
                let a=(y*N+x) as u16;assert!(s.legal_moves().contains(&a));
                s.apply_move_in_place_no_undo(a);
            }
            assert!(s.is_terminal());assert_eq!(s.outcome(),-1.);
        }}
    }
}
