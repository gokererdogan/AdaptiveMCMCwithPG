- I don't need to calculate the derivative of the pieces of MH transition kernel
(propose, accept/reject) and combine these myself. I can just wrap the MH kernel
into a single function and take its derivative using autograd.
-- However, this would not give the incorrect derivative for the reject case. In fact,
it would always give zero because the next state does not depend on the parameters if
it is rejected.