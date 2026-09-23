"""Local smoke tests; camera testing is intentionally local-only."""
import sys,cv2
def test_display():
    from core.display import build_virtual_desktop,build_trackpad_zone
    d=build_virtual_desktop(); z=build_trackpad_zone()
    assert d.total_width>0 and d.total_height>0 and z.x_max>z.x_min and z.y_max>z.y_min
def test_camera():
    from config import CAMERA_INDEX
    cap=cv2.VideoCapture(CAMERA_INDEX,cv2.CAP_DSHOW)
    try:
        assert cap.isOpened(),f"Camera {CAMERA_INDEX} failed to open"
        ok,frame=cap.read(); assert ok and frame is not None and frame.size>0
    finally: cap.release()
def test_mediapipe():
    from core.tracker import HandTracker
    t=HandTracker(); t.close()
def test_orchestrator():
    from core.display import build_virtual_desktop,build_trackpad_zone
    from core.actuator import MouseActuator
    from core.gestures import GestureOrchestrator
    from core.tracker import HandsResult
    d=build_virtual_desktop(); z=build_trackpad_zone(); a=MouseActuator(d.total_width,d.total_height)
    GestureOrchestrator(a,d,z).process(HandsResult())
def main():
    tests=[("Display",test_display),("Camera",test_camera),("MediaPipe",test_mediapipe),("Orchestrator",test_orchestrator)]
    passed=0
    for name,test in tests:
        try:test();print("[PASS]",name);passed+=1
        except Exception as exc:print("[FAIL]",name,exc)
    print(f"\n{passed}/{len(tests)} local tests passed")
    sys.exit(0 if passed==len(tests) else 1)
if __name__=="__main__":main()
