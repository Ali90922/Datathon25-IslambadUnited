import { Link } from "react-router-dom";

const Header = () => {
	return (
		<nav className='bg-linear-to-b from-slate-800 to-background'>
			<ul className='flex items-center justify-center'>
				<Link to='/' className='text-3xl font-bold p-8'>
					Substance Sense
				</Link>
			</ul>
		</nav>
	);
};

export default Header;
