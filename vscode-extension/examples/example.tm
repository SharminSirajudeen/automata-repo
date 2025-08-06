{
  "states": ["q0", "q1", "q2", "qaccept", "qreject"],
  "alphabet": ["0", "1"],
  "tape_alphabet": ["0", "1", "X", "Y", "_"],
  "blank_symbol": "_",
  "start_state": "q0",
  "accept_states": ["qaccept"],
  "reject_states": ["qreject"],
  "transitions": {
    "q0": {
      "0": {
        "state": "q1",
        "write": "X",
        "move": "R"
      },
      "1": {
        "state": "qreject",
        "write": "1",
        "move": "R"
      },
      "_": {
        "state": "qaccept",
        "write": "_",
        "move": "R"
      },
      "Y": {
        "state": "q0",
        "write": "Y",
        "move": "R"
      }
    },
    "q1": {
      "0": {
        "state": "q1",
        "write": "0",
        "move": "R"
      },
      "1": {
        "state": "q2",
        "write": "Y",
        "move": "L"
      },
      "_": {
        "state": "qreject",
        "write": "_",
        "move": "R"
      },
      "Y": {
        "state": "q1",
        "write": "Y",
        "move": "R"
      }
    },
    "q2": {
      "0": {
        "state": "q2",
        "write": "0",
        "move": "L"
      },
      "X": {
        "state": "q0",
        "write": "X",
        "move": "R"
      },
      "Y": {
        "state": "q2",
        "write": "Y",
        "move": "L"
      }
    }
  }
}