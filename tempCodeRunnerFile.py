
    # QmaxDia = float(QMaxDia.get())
    # QrelVel = float(QRelVel.get())
    # QmisDis = float(QMisDis.get())
    # cat_MaxDia = "Large" if QmaxDia >= max_dia_mean else "Small"
    # cat_RelVel = "Fast" if QrelVel >= Rv_mean else "Slow"
    # cat_MisDis = "Less" if QmisDis >= 0 and QmisDis < 25000000 else ("Medium" if QmisDis>= 25000000 and QmisDis<50000000 else "More")
    # test_record = pd.DataFrame(list(zip(QmaxDia, QrelVel, QmisDis, cat_MaxDia, cat_RelVel, cat_MisDis, "True")), columns=['Max_Diameter','Relative_Velocity','Miss_Distance', 'Categorized_Diameter', 'Categorized_Relative_Vel','Categorised_Miss_Distance','Hazardous'])
    # print(test_record.iloc[0])
    # QdisplayValue.set(str(True))