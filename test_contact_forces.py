model_xml = """<mujoco model="test">
    <worldbody>
        <body name="wall" pos="0 0 0" xyaxes="1 0 0 0 0 1">
        <geom name="rwall2" pos = "0 2 -2" size=".1 2 .05" type="plane"
            material="grid" condim="3"/>
        </body>
    </worldbody>
</mujoco>"""
