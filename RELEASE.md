# E177 engineering transport control (2026-11-13)

This package retains exact E132 solver, scheduler, prompts and executor. It adds the E170 streaming transport and the E132-style output usage whitelist. The scoped executor from E171/E175 is excluded. The E171/E175 comparative quality gate failed and has not been waived; this control-only update makes no quality improvement claim. The seven gateway assertions remain unchanged, with a streaming-aware test response fixture.

Source freeze SHA-256: `89eb175ca77eee0066fc01bf8879b8c0640efd84f69ccfe86afbdcfe67d27280`. Four completed E170 controls, offline actual-wire replay and native SDK mocks are technical transport evidence, not an official score. E132 async public-test compatibility limitation remains.
