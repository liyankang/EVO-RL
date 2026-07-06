python teleop_service/run.py --sim-mj --host 0.0.0.0 --port 5000


python teleop_service/run.py --real-piper --no-calibrate



python teleop_service/phone_server.py --can-port can0 --no-calibrate



    使用 Piper 真机（CAN 总线）
Traceback (most recent call last):
  File "/home/user/code_1/evo_rl/teleop_service/run.py", line 328, in <module>
    main()
  File "/home/user/code_1/evo_rl/teleop_service/run.py", line 248, in main
    robot = PiperRealRobot(
TypeError: PiperRealRobot.__init__() got an unexpected keyword argument 'ik_solver'
