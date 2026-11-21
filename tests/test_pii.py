from sentineldesk.pii import mask_pii


def test_plan_example():
    m, kinds = mask_pii("Mail a.b@x.com, call +49 170 1234567, IP 10.0.0.1, IBAN DE89370400440532013000, see TKT-00042")
    assert "a.b@x.com" not in m and "10.0.0.1" not in m and "DE89370400440532013000" not in m
    assert "170 1234567" not in m
    assert {"EMAIL", "PHONE", "IP", "IBAN"} <= set(kinds) and "TKT-00042" in m


def test_demo_ticket():
    t = ("I cant login! My username is traxy122@zmail.com\n"
         "My billing address is Sesame Street 4, MA 02120.\n"
         "My contact number is +1 1112223331\nPlease help asap")
    m, kinds = mask_pii(t)
    assert "traxy122@zmail.com" not in m and "1112223331" not in m
    assert "[EMAIL]" in m and "[PHONE]" in m and set(kinds) == {"EMAIL", "PHONE"}
    assert "MA 02120" in m  # postcodes stay; addresses need NER, not regex


def test_german_formats_and_iban_with_spaces():
    m, kinds = mask_pii("Rufen Sie 0170/1234567 oder 030 12345678 an. IBAN: DE89 3704 0044 0532 0130 00")
    assert "1234567" not in m and "12345678" not in m and "3704" not in m
    assert {"PHONE", "IBAN"} <= set(kinds)


def test_card_and_account():
    m, kinds = mask_pii("Card 4111 1111 1111 1111, account no. 12345678, Kundennummer: 9876543")
    assert "4111" not in m and "12345678" not in m and "9876543" not in m
    assert {"CARD", "ACCOUNT"} <= set(kinds)


def test_ipv6():
    m, kinds = mask_pii("server at 2001:db8:85a3::8a2e:370:7334 is down")
    assert "2001:db8" not in m and kinds == ["IP"]


def test_leaves_normal_text_alone():
    t = ("VPN down since 08:45 on 21.11.2026, client 2.4.1, ticket TKT-00042, "
         "error 503, 200 users, order 12345")
    m, kinds = mask_pii(t)
    assert m == t and kinds == []


def test_empty():
    assert mask_pii("") == ("", [])
    assert mask_pii(None) == ("", [])
