from lewm.continuous_start_connector_development import propose,grid_propose


def test_clear_disk_can_leave_conservatively_inflated_start_cell():
    occupied={(0,11)}
    floor={(x,y) for x in range(-3,4) for y in range(-4,2)}
    start=[.025,.06];goal=[.025,-.15]
    assert grid_propose(floor,occupied,start,goal)['status']=='ADDITIONAL_VIEW_REQUIRED'
    result=propose(floor,occupied,start,goal)
    assert result['route_cells']
    assert result['nominal_radius_m']==.45
    assert not result['motion_permitted']
    # Actual disk contact remains rejected, including the tangency boundary.
    assert propose(floor,occupied,[.025,.10],goal)['status']=='ADDITIONAL_VIEW_REQUIRED'
    assert propose(floor,occupied,[.025,.11],goal)['status']=='ADDITIONAL_VIEW_REQUIRED'
    calls=[]
    def blocked(start,end,cells,radius):
        calls.append((list(start),list(end)))
        return False
    result=grid_propose(floor,occupied,[.025,.11],goal,connector_clear=blocked)
    assert result['status']=='ADDITIONAL_VIEW_REQUIRED'
    assert calls==[([.025,.11],[.025,.11])]
