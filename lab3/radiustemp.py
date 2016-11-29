
l0 = []
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
l7 = []
l8 = []
l9 = []
l10 = []
l11 = []

l0e = []
l1e = []
l2e = []
l3e = []
l4e = []
l5e = []
l6e = []
l7e = []
l8e = []
l9e = []
l10e = []
l11e = []

for ok in np.arange(len(timefromstart))[0:len(bin0)]:
    l0.append(wfux[ok])
    l0e.append(werrorfux[ok] ** 2)
hill0 = sum(l0) / len(bin0)
hill0e = (sum(l0e)) ** .5 / len(bin0)

for ok in np.arange(len(timefromstart))[len(bin0):len(bin0) + len(bin1)]:
    l1.append(wfux[ok])
    l1e.append(werrorfux[ok] ** 2)
hill1 = sum(l1) / len(bin1)
hill1e = (sum(l1e)) ** .5 / len(bin1)

for ok in np.arange(len(timefromstart))[len(bin0) + len(bin1):len(bin0) + len(bin1) + len(bin2)]:
    l2.append(wfux[ok])
    l2e.append(werrorfux[ok] ** 2)
hill2 = sum(l2) / len(bin2)
hill2e = (sum(l2e)) ** .5 / len(bin2)

for ok in np.arange(len(timefromstart))[
          len(bin0) + len(bin1) + len(bin2):len(bin0) + len(bin1) + len(bin2) + len(bin3)]:
    l3.append(wfux[ok])
    l3e.append(werrorfux[ok] ** 2)
hill3 = sum(l3) / len(bin3)
hill3e = (sum(l3e)) ** .5 / len(bin3)

for ok in np.arange(len(timefromstart))[
          len(bin0) + len(bin1) + len(bin2) + len(bin3):len(bin0) + len(bin1) + len(bin2) + len(bin3) + len(bin4)]:
    l4.append(wfux[ok])
    l4e.append(werrorfux[ok] ** 2)
hill4 = sum(l4) / len(bin4)
hill4e = (sum(l4e)) ** .5 / len(bin4)

for ok in np.arange(len(timefromstart))[
          len(bin0) + len(bin1) + len(bin2) + len(bin3) + len(bin4):len(bin0) + len(bin1) + len(bin2) + len(
              bin3) + len(
              bin4) + len(bin5)]:
    l5.append(wfux[ok])
    l5e.append(werrorfux[ok] ** 2)
hill5 = sum(l5) / len(bin5)
hill5e = (sum(l5e)) ** .5 / len(bin5)

for ok in np.arange(len(timefromstart))[
          len(bin0) + len(bin1) + len(bin2) + len(bin3) + len(bin4) + len(bin5):len(bin0) + len(bin1) + len(
              bin2) + len(
              bin3) + len(bin4) + len(bin5) + len(bin6)]:
    l6.append(wfux[ok])
    l6e.append(werrorfux[ok] ** 2)
hill6 = sum(l6) / len(bin6)
hill6e = (sum(l6e)) ** .5 / len(bin6)

for ok in np.arange(len(timefromstart))[
          len(bin0) + len(bin1) + len(bin2) + len(bin3) + len(bin4) + len(bin5) + len(bin6):len(bin0) + len(
              bin1) + len(
              bin2) + len(bin3) + len(bin4) + len(bin5) + len(bin6) + len(bin7)]:
    l7.append(wfux[ok])
    l7e.append(werrorfux[ok] ** 2)
hill7 = sum(l7) / len(bin7)
hill7e = (sum(l7e)) ** .5 / len(bin7)

for ok in np.arange(len(timefromstart))[
          len(bin0) + len(bin1) + len(bin2) + len(bin3) + len(bin4) + len(bin5) + len(bin6) + len(bin7):len(
              bin0) + len(
              bin1) + len(bin2) + len(bin3) + len(bin4) + len(bin5) + len(bin6) + len(bin7) + len(bin8)]:
    l8.append(wfux[ok])
    l8e.append(werrorfux[ok] ** 2)
hill8 = sum(l8) / len(bin8)
hill8e = (sum(l8e)) ** .5 / len(bin8)

for ok in np.arange(len(timefromstart))[
          len(bin0) + len(bin1) + len(bin2) + len(bin3) + len(bin4) + len(bin5) + len(bin6) + len(bin7) + len(
              bin8):len(
              bin0) + len(bin1) + len(bin2) + len(bin3) + len(bin4) + len(bin5) + len(bin6) + len(bin7) + len(
              bin8) + len(bin9)]:
    l9.append(wfux[ok])
    l9e.append(werrorfux[ok] ** 2)
hill9 = sum(l9) / len(bin9)
hill9e = (sum(l9e)) ** .5 / len(bin9)

for ok in np.arange(len(timefromstart))[
          len(bin0) + len(bin1) + len(bin2) + len(bin3) + len(bin4) + len(bin5) + len(bin6) + len(bin7) + len(
              bin8) + len(bin9):len(bin0) + len(bin1) + len(bin2) + len(bin3) + len(bin4) + len(bin5) + len(
              bin6) + len(bin7) + len(bin8) + len(bin9) + len(bin10)]:
    l10.append(wfux[ok])
    l10e.append(werrorfux[ok] ** 2)
hill10 = sum(l10) / len(bin10)
hill10e = (sum(l10e)) ** .5 / len(bin10)

for ok in np.arange(len(timefromstart))[
          len(bin0) + len(bin1) + len(bin2) + len(bin3) + len(bin4) + len(bin5) + len(bin6) + len(bin7) + len(
              bin8) + len(bin9) + len(bin10):len(bin0) + len(bin1) + len(bin2) + len(bin3) + len(bin4) + len(
              bin5) + len(bin6) + len(bin7) + len(bin8) + len(bin9) + len(bin10) + len(bin11)]:
    l11.append(wfux[ok])
    l11e.append(werrorfux[ok] ** 2)
hill11 = sum(l11) / len(bin11)
hill11e = (sum(l11e)) ** .5 / len(bin11)

pilly = [hill0, hill1, hill2, hill3, hill4, hill5, hill6, hill7, hill8, hill9, hill10, hill11]
pillye = [hill0e, hill1e, hill2e, hill3e, hill4e, hill5e, hill6e, hill7e, hill8e, hill9e, hill10e, hill11e]
