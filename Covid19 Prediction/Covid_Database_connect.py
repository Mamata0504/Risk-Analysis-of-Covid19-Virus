import sqlite3


def insert_record(sex,pneumonia,age,pregnancy,diabetes,copd,asthma,inmsupr,hypertension,other_disease,cardiovascular,obesity,renal_chronic,contact_other_covid,output):
    
    try:
        con =sqlite3.connect('Covid19_data.db')
        cur =con.cursor()

        cur.execute(""" CREATE TABLE IF NOT EXISTS customer(
                                        sex integer Not Null,
                                        pneumonia integer Not Null,
                                        age integer Not Null,
                                        pregnancy integer Not Null,
                                        diabetes integer Not Null,
                                        copd integer Not Null,
                                        asthma integer Not Null,
                                        inmsupr integer Not Null,
                                        hypertension integer Not Null,
                                        other_disease integer Not Null,
                                        cardiovascular integer Not Null,
                                        obesity integer Not Null,
                                        renal_chronic integer Not Null,
                                        contact_other_covid integer Not Null,
                                        Covid19_status integer Not Null
                                    ); """)
        #rec=[feat for x in record for feat in x]
        cur.execute("""INSERT INTO customer(sex,pneumonia,age,pregnancy,diabetes,copd,asthma,inmsupr,hypertension,other_disease,cardiovascular,obesity,renal_chronic,contact_other_covid,Covid19_status)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", 
                        (sex,pneumonia,age,pregnancy,diabetes,copd,asthma,inmsupr,hypertension,other_disease,cardiovascular,obesity,renal_chronic,contact_other_covid,output))
        print("Record Inserted")
    except Exception as e:
        print("Unable to insert",e)   
    con.commit()
    con.close()