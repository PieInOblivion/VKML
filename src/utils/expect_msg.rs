pub trait ExpectMsg<T, E> {
    fn expect_msg(self, msg: &str) -> T;
}

impl<T, E: std::fmt::Debug> ExpectMsg<T, E> for Result<T, E> {
    fn expect_msg(self, msg: &str) -> T {
        match self {
            Ok(val) => val,
            Err(err) => panic!("{}: {:?}", msg, err),
        }
    }
}
