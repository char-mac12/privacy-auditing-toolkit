# privacy-auditing-toolkit
venv\Scripts\activate (windows)
source venv/bin/activate (mac/linux)

pip install -r requirements.txt

python run.py --config configs/pythia_160m_pile_cc.json
python run.py --config configs/pythia_160m_wikitext.json